"""
Live mirror of a device's screen as decoded BGRA frames.

:class:`ScreenMirrorSession` owns the device side of screen mirroring: it starts the
CoreDevice video stream (and optionally the paired audio stream, played on the host's
speakers), depacketizes the RTP/HEVC stream, decodes it to BGRA, and keeps the session
alive and recoverable with RTCP keepalives and keyframe requests. Decoded frames reach
the embedder through ``on_frame`` (on the event-loop thread) and :attr:`latest_frame`.

Pipeline: RTP/HEVC -> HEVC decoder (VideoToolbox or libav, BGRA output) -> ``on_frame``.

The VNC server (:mod:`pymobiledevice3.remote.core_device.vnc_server`) is one consumer,
serving the frames over RFB.
"""

import asyncio
import contextlib
import logging
import struct
import sys
import uuid
from collections import deque
from collections.abc import Awaitable
from types import TracebackType
from typing import Any, Callable, Optional

from pymobiledevice3.exceptions import CoreDeviceError, ScreenMirrorStopped
from pymobiledevice3.remote.core_device.aac_eld import AACELDDecoder
from pymobiledevice3.remote.core_device.audio_player import AudioQueuePlayer
from pymobiledevice3.remote.core_device.display_service import (
    MEDIA_IN_USE_MESSAGE,
    DisplayService,
    is_media_in_use_error,
)
from pymobiledevice3.remote.core_device.hevc_rps import HevcRpsTracker, is_slice_nal
from pymobiledevice3.remote.core_device.hid_service import (
    DIGITIZER_SURFACE_MAIN_TOUCHSCREEN,
    HID_BUTTON_STATE_DOWN,
    HID_BUTTON_STATE_UP,
    TOUCHSCREEN_STATE_CONTACT,
    TOUCHSCREEN_STATE_RELEASE,
    IndigoHIDService,
    UniversalHIDServiceService,
)
from pymobiledevice3.remote.core_device.screen_stream import UdpMediaTransport, depacketize_hevc, open_media_receiver
from pymobiledevice3.remote.remote_service_discovery import RemoteServiceDiscoveryService

logger = logging.getLogger(__name__)

# Teardown bounds for :meth:`ScreenMirrorSession.stop`, which must return even when the device
# is gone and every device call hangs: one step closes the HID and stream connections
# concurrently, the next releases the device-side media session. Together they stay < 10 s.
STOP_STEP_TIMEOUT = 3.0
STOP_ALL_STREAMS_TIMEOUT = 6.0

# ``(bgra, width, height)`` of a decoded frame; called on the event-loop thread.
FrameCallback = Callable[[bytes, int, int], None]


def _consume_task_result(task: "asyncio.Task[Any]") -> None:
    if not task.cancelled():
        task.exception()  # mark retrieved: a late failure of an abandoned call is expected


async def _await_bounded(aw: Awaitable[Any], timeout: float, what: str) -> None:
    """Await ``aw`` for at most ``timeout`` seconds, logging (not raising) a timeout or failure.

    Unlike ``asyncio.wait_for`` (before Python 3.12), a timed-out call is cancelled but not
    waited for, so a hung device call that does not honour cancellation cannot block teardown.
    """
    task = asyncio.ensure_future(aw)
    task.add_done_callback(_consume_task_result)
    try:
        done, _ = await asyncio.wait({task}, timeout=timeout)
    except asyncio.CancelledError:
        task.cancel()
        raise
    if not done:
        task.cancel()
        logger.warning("shutdown: %s did not finish within %.1f s", what, timeout)
    elif not task.cancelled() and task.exception() is not None:
        logger.debug("shutdown: %s failed", what, exc_info=task.exception())


def _resolve_decoder(choice: str):
    """Pick the HEVC decoder class.

    ``choice`` is one of ``"auto"`` (VideoToolbox on macOS, libav
    everywhere else), ``"vt"`` (force VideoToolbox), or ``"av"``
    (force libav / PyAV).

    Both classes expose the same ``feed`` / ``on_frame`` /
    ``on_decode_error`` / ``width`` / ``height`` / ``close`` surface so
    the recv loop below doesn't branch on the choice.
    """
    choice = (choice or "auto").lower()
    if choice == "auto":
        choice = "vt" if sys.platform == "darwin" else "av"
    if choice == "vt":
        from pymobiledevice3.remote.core_device.vt_jpeg import HevcToBgraTranscoder

        return HevcToBgraTranscoder
    if choice == "av":
        from pymobiledevice3.remote.core_device.hevc_av import HevcToBgraTranscoder

        return HevcToBgraTranscoder
    raise ValueError(f"unknown decoder choice: {choice!r}")


_HEVC_NAL_IDR_W_RADL = 19
_HEVC_NAL_IDR_N_LP = 20
_HEVC_NAL_CRA = 21
_HEVC_NAL_VPS = 32
_HEVC_NAL_SPS = 33
_HEVC_NAL_PPS = 34


def _is_key_nal(nt: int) -> bool:
    return nt in (_HEVC_NAL_IDR_W_RADL, _HEVC_NAL_IDR_N_LP, _HEVC_NAL_CRA)


# Named iOS hardware buttons → (usage_page, usage_code). Same table as
# ``cli/developer/core_device.py``; replicated rather than imported so
# this server module stays decoupled from the CLI layer.
_BTN_HOME = (0x0C, 0x40)
_BTN_LOCK = (0x0C, 0x30)
_BTN_VOL_UP = (0x0C, 0xE9)
_BTN_VOL_DOWN = (0x0C, 0xEA)
_BTN_MUTE = (0x0C, 0xE2)
_BTN_SIRI = (0x0C, 0xCF)

# Public name -> (usage_page, usage_code) table for :meth:`ScreenMirrorSession.send_button`.
HARDWARE_BUTTONS: dict[str, tuple[int, int]] = {
    "home": _BTN_HOME,
    "lock": _BTN_LOCK,
    "volume-up": _BTN_VOL_UP,
    "volume-down": _BTN_VOL_DOWN,
    "mute": _BTN_MUTE,
    "siri": _BTN_SIRI,
}


class ScreenMirrorSession:
    """Mirror the device screen as decoded BGRA frames.

    ``start()`` brings the device stream up and returns once it is negotiated; frames then
    arrive on ``on_frame`` (and in :attr:`latest_frame`) until ``stop()``. Usable as an
    async context manager.
    """

    def __init__(
        self,
        rsd: RemoteServiceDiscoveryService,
        *,
        display_id: int = 1,
        audio: bool = False,
        decoder: str = "auto",
        allow_rtcp_fb: bool = False,
        ltrp_enabled: bool = False,
        on_frame: Optional[FrameCallback] = None,
    ) -> None:
        self._rsd = rsd
        self._display_id = display_id
        self._audio_enabled = audio
        self._sender_ip = rsd.service.address[0]
        # Protobuf-level negotiation knobs; see media_stream_offer.py.
        self._allow_rtcp_fb = allow_rtcp_fb
        self._ltrp_enabled = ltrp_enabled
        self._transcoder_cls = _resolve_decoder(decoder)
        logger.info("HEVC decoder: %s", self._transcoder_cls.__module__)

        # Filled once we have a decoder + first frame.
        self._fb_width = 0
        self._fb_height = 0
        self._ready = asyncio.Event()
        # Raw BGRA bytes of the most recent decoded frame. Exactly
        # ``fb_width * fb_height * 4`` bytes once ``_ready`` is set.
        self._latest_frame: Optional[bytes] = None

        self._transcoder = None  # type: ignore[assignment]
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._frames_emitted = 0

        # RTCP feedback (PLI = "send me a fresh IDR"). Populated from
        # ``streamConfig`` after start_video_stream returns. Without
        # these we can't kick the encoder, and decoder-refresh becomes
        # a no-op.
        self._active_transport: Optional[UdpMediaTransport] = None
        self._local_ssrc: int = 0
        self._remote_ssrc: int = 0
        self._rtcp_dest: Optional[tuple[str, int]] = None
        self._pli_tasks: set[asyncio.Task[None]] = set()
        # Extended highest video RTP seq, for the periodic Receiver Report that
        # keeps the device from reaping the video session (RTCPTimeoutEnabled).
        self._rtp_highest_seq: int = 0

        # Motion / decode-error recovery: on a byte-rate motion spike, request
        # a keyframe (RTCP PLI, or FIR when --rtcp-fb is set) and rebuild the
        # transcoder from the next IDR. NOTE: this does not resolve the
        # heavy-motion smear -- that is a receiver-side decode issue, not
        # recoverable from here (see VIDEOPROCESSING_TEAR_FINDINGS.md).
        self._au_byte_window: deque[tuple[float, int]] = deque()
        self._motion_active = False
        self._motion_ended_t: float = 0.0
        self._last_refresh_t: float = 0.0
        # While True, stop feeding deltas and rebuild the transcoder on the
        # next IDR (so no more state accumulates while waiting for it).
        self._refresh_pending = False

        # Sticky keyframe-required state: set on a decode error, cleared only
        # when a fresh IDR has been fed AND a clean frame decoded after it, so
        # a lost/corrupt IDR response to a keyframe request does not leave the
        # decoder stuck on stale references.
        self._keyframe_required = False
        self._idr_observed = False
        # Reference-picture-set tracker: flags a P-slice that references a POC
        # we never decoded, to fire an early keyframe request on that signal.
        self._rps_tracker = HevcRpsTracker()
        # Grace window after feeding an IDR: pre-IDR P-frames still queued in
        # VT error against the reset reference set -- expected drain noise.
        self._idr_observed_at: float = 0.0
        self._fir_attempts: int = 0
        self._cap_warned: bool = False

        # Audio state -- mirrors screen_stream.py's audio side.
        # Decoded PCM goes straight to the host's speakers via the
        # AudioQueuePlayer; nothing is broadcast over RFB (RFB has no
        # audio of its own). The audio session is paired with the
        # video session on the device by sharing client_session_id.
        self._audio_player: Optional[AudioQueuePlayer] = None
        self._audio_decoder: Optional[AACELDDecoder] = None
        self._audio_transport: Optional[UdpMediaTransport] = None
        self._audio_local_ssrc: int = 0
        self._audio_remote_ssrc: int = 0
        self._audio_rtcp_dest: Optional[tuple[str, int]] = None
        self._audio_rtp_packets_received: int = 0
        self._audio_rtp_highest_seq: int = 0
        self._audio_session_id: Optional[uuid.UUID] = None
        self._audio_svc: Optional[DisplayService] = None

        # HID for pointer-event translation (UniversalHIDService for the
        # touchscreen surface, IndigoHIDService for hardware buttons
        # like Home / Lock / Volume / Siri / Mute).
        self._uhs: Optional[UniversalHIDServiceService] = None
        self._indigo: Optional[IndigoHIDService] = None
        self._hid_lock = asyncio.Lock()
        # _ServiceID dtuhidd assigned to our host-registered virtual
        # keyboard. Lazily filled on first key event; serialized by
        # ``_hid_lock`` so a burst of keystrokes at startup doesn't race
        # multiple createService calls.
        self._kb_service_id: Optional[int] = None

        self._on_frame = on_frame
        # What ``start()`` created, for ``stop()`` to tear down. The session is single-use:
        # ``_start_called`` / ``_stopped`` never reset.
        self._start_called = False
        self._started = False  # the device video stream is up
        self._stream_requested = False  # start_video_stream was sent (the device may hold a session)
        self._stopped = False
        # Set when an in-flight ``start()`` has unwound (or never ran), for ``stop()`` to wait on.
        self._start_exited = asyncio.Event()
        self._start_exited.set()
        self._start_task: Optional[asyncio.Task[Any]] = None
        # Set by ``stop()`` when it cancels the in-flight ``start()``, so ``start()`` can raise
        # ScreenMirrorStopped for that cancellation (and only that one).
        self._cancelled_by_stop = False
        # ``wait_stopped()``: set by ``stop()`` (or a failed ``start()``) and by the first
        # failure of the video feed / RTCP task, which ``_failure`` then records.
        self._stopped_event = asyncio.Event()
        self._failure: Optional[BaseException] = None
        # Set once the single teardown (by stop() or a failed start()) has finished; every
        # stop() after the first awaits it.
        self._teardown_done = asyncio.Event()
        self._svc: Optional[DisplayService] = None
        self._transport: Optional[UdpMediaTransport] = None
        self._feed_task: Optional[asyncio.Task[None]] = None
        self._refresh_task: Optional[asyncio.Task[None]] = None
        self._rtcp_task: Optional[asyncio.Task[None]] = None
        self._audio_recv_task: Optional[asyncio.Task[None]] = None
        self._audio_rtcp_task: Optional[asyncio.Task[None]] = None

    # ----- public API ---------------------------------------------------------
    @property
    def frame_size(self) -> tuple[int, int]:
        """``(width, height)`` of the decoded frames; ``(0, 0)`` before the first frame."""
        return self._fb_width, self._fb_height

    @property
    def latest_frame(self) -> Optional[bytes]:
        """Raw BGRA bytes of the most recent decoded frame, or ``None`` before the first one."""
        return self._latest_frame

    async def wait_first_frame(self) -> None:
        """Wait until the first frame is decoded (and :attr:`frame_size` is known)."""
        await self._ready.wait()

    async def wait_stopped(self) -> Optional[BaseException]:
        """Wait until the session is over.

        Returns ``None`` once :meth:`stop` has completed (or a failed :meth:`start` has unwound),
        or the exception the video feed or its RTCP keepalive ended with -- e.g. an ``OSError``
        when the device was unplugged. After a failure the session still needs :meth:`stop`.
        """
        await self._stopped_event.wait()
        return self._failure

    def _on_stream_task_done(self, task: "asyncio.Task[None]") -> None:
        """Done-callback of the video feed / RTCP tasks: report the first one that dies."""
        if task.cancelled():
            return
        exc = task.exception()
        if exc is None:
            return
        logger.warning("screen-mirror stream ended: %r", exc)
        if self._failure is None:
            self._failure = exc
        self._stopped_event.set()

    async def __aenter__(self) -> "ScreenMirrorSession":
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> None:
        await self.stop()

    def _consumer_count(self) -> int:
        """How many viewers want decoded frames; the decoder-refresh loop idles at zero.
        The session itself always consumes them (``latest_frame`` / ``on_frame``)."""
        return 1

    # ----- HEVC -> BGRA callback marshalling --------------------------------
    def _on_frame_from_worker(self, bgra: bytes) -> None:
        """Called from the VT transcoder worker thread."""
        loop = self._loop
        if loop is None:
            return
        loop.call_soon_threadsafe(self._broadcast_frame, bgra)

    def _on_decode_error_from_worker(self) -> None:
        """Worker-thread hook for VT decode failures / FrameDropped."""
        loop = self._loop
        if loop is None:
            return
        loop.call_soon_threadsafe(self._on_decode_error)

    def _on_decode_error(self) -> None:
        """Loop-thread handler. Promotes a decode error into the sticky
        keyframe-required state and arms a refresh, subject to the
        post-IDR grace window and the existing ``_refresh_pending``
        gate (don't double-arm an in-flight recovery)."""
        loop = self._loop
        if loop is None:
            return
        now = loop.time()
        # Post-IDR grace: errors right after we fed an IDR are almost
        # always queued pre-IDR P-frames decoding against the freshly
        # reset reference set. Suppress them so they don't trigger
        # back-to-back PLI storms. 500 ms covers the typical drain.
        if self._idr_observed_at > 0.0 and (now - self._idr_observed_at) < 0.5:
            return
        if self._refresh_pending:
            return  # already armed; let the IDR land first
        if self._keyframe_required:
            # Already in recovery -- the periodic re-arm in
            # _decoder_refresh_loop handles further PLI attempts.
            return
        self._keyframe_required = True
        self._idr_observed = False
        self._fire_decoder_refresh(now, reason="decode-error")

    def _broadcast_frame(self, bgra: bytes) -> None:
        self._latest_frame = bgra
        self._frames_emitted += 1
        if not self._ready.is_set() and self._transcoder is not None:
            self._fb_width = self._transcoder.width
            self._fb_height = self._transcoder.height
            self._ready.set()
        # Sticky-keyframe two-condition clear: a clean decoded frame
        # only counts as "recovered" if we already saw the IDR fed
        # for this recovery cycle. Without `_idr_observed` we'd
        # false-clear on a pre-IDR P-frame that happened to decode
        # without an API error.
        if self._keyframe_required and self._idr_observed:
            self._keyframe_required = False
            self._idr_observed = False
            self._fir_attempts = 0
            self._cap_warned = False
            logger.debug("decode recovered (IDR + clean frame)")
        if self._on_frame is not None:
            self._on_frame(bgra, self._fb_width, self._fb_height)

    # ----- RTP recv + transcoder feed ---------------------------------------
    async def _udp_recv_and_pipe(self, transport: UdpMediaTransport) -> None:
        """Same depacketize loop as ScreenStreamServer: gather Annex-B
        AUs and feed the VT transcoder.

        A socket-level receive error (``OSError``, e.g. the device went away) ends the task
        with that exception, which :meth:`wait_stopped` reports."""
        loop = asyncio.get_running_loop()
        fu_buffer = bytearray()
        current_au: list[bytes] = []
        last_seq: Optional[int] = None
        au_corrupt = False
        au_is_key = False
        nals: list[bytes] = []
        cached_vps: Optional[bytes] = None
        cached_sps: Optional[bytes] = None
        cached_pps: Optional[bytes] = None
        rtp_packets = 0
        feed_count = 0
        frame_count_at_last_log = 0
        last_stat_t = loop.time()
        while True:
            try:
                data = await transport.recv()
            except asyncio.CancelledError:
                return
            if len(data) < 12:
                continue
            pt = data[1] & 0x7F
            if 64 <= pt <= 95:
                continue
            marker = (data[1] >> 7) & 1
            cc = data[0] & 0x0F
            header_len = 12 + cc * 4
            if data[0] & 0x10:
                ext_len = int.from_bytes(data[header_len + 2 : header_len + 4], "big")
                header_len += 4 + ext_len * 4
            payload = data[header_len:]

            seq = int.from_bytes(data[2:4], "big")
            # Maintain the extended highest-seq for the RR keepalive.
            cur_ext = self._rtp_highest_seq
            cycles = (cur_ext >> 16) & 0xFFFF
            last_seq16 = cur_ext & 0xFFFF
            if seq < last_seq16 and (last_seq16 - seq) > 0x8000:
                cycles = (cycles + 1) & 0xFFFF
            new_ext = (cycles << 16) | seq
            if cur_ext == 0 or ((new_ext - cur_ext) & 0xFFFFFFFF) < 0x80000000:
                self._rtp_highest_seq = new_ext
            if last_seq is not None and seq != ((last_seq + 1) & 0xFFFF):
                fu_buffer.clear()
                au_corrupt = True
            if last_seq is None or ((seq - last_seq) & 0xFFFF) < 0x8000:
                last_seq = seq

            nals.clear()
            depacketize_hevc(payload, fu_buffer, nals)
            for nal in nals:
                if not nal:
                    continue
                nt = (nal[0] >> 1) & 0x3F
                if nt == _HEVC_NAL_VPS:
                    cached_vps = bytes(nal)
                elif nt == _HEVC_NAL_SPS:
                    cached_sps = bytes(nal)
                    self._rps_tracker.feed_sps(cached_sps)
                elif nt == _HEVC_NAL_PPS:
                    cached_pps = bytes(nal)
                elif _is_key_nal(nt):
                    au_is_key = True
                current_au.append(nal)

            if marker:
                if current_au and not au_corrupt:
                    # Motion signal: track AU sizes in a 1 s window so
                    # the refresh loop can detect motion start/end.
                    au_bytes = sum(len(n) for n in current_au)
                    self._au_byte_window.append((loop.time(), au_bytes))
                    # Decoder rebuild on a fresh IDR -- the previous
                    # transcoder's LTRP table is suspected stale, so we
                    # close it and let the bootstrap branch below build
                    # a fresh one from this very IDR.
                    if self._refresh_pending and au_is_key and self._transcoder is not None:
                        with contextlib.suppress(Exception):
                            self._transcoder.close()
                        self._transcoder = None
                        self._refresh_pending = False
                        # Drop the RPS DPB shadow with the decoder: a rebuilt
                        # VT only has the next IDR's POC in its DPB, so keeping
                        # the old seen-set would let a P-slice referencing a
                        # pre-rebuild POC slip past the pre-decode check.
                        self._rps_tracker.reset()
                        # Sticky-keyframe IDR observation: the new
                        # transcoder built below will be seeded by
                        # this very IDR. Stamp the grace window now
                        # so the new VT's first few frames aren't
                        # treated as fresh decode errors, and arm
                        # the IDR-observed half of the two-condition
                        # clear if recovery was in progress.
                        self._idr_observed_at = loop.time()
                        if self._keyframe_required:
                            self._idr_observed = True
                        logger.debug("decoder rebuilt on fresh IDR")
                    if (
                        self._transcoder is None
                        and au_is_key
                        and cached_vps is not None
                        and cached_sps is not None
                        and cached_pps is not None
                    ):
                        try:
                            self._transcoder = self._transcoder_cls(
                                cached_vps,
                                cached_sps,
                                cached_pps,
                                on_frame=self._on_frame_from_worker,
                                on_decode_error=self._on_decode_error_from_worker,
                            )
                            logger.info(
                                "VT transcoder started: HEVC %dx%d -> BGRA",
                                self._transcoder.width,
                                self._transcoder.height,
                            )
                        except Exception:
                            logger.exception("VT transcoder failed to start")
                    # While ``_refresh_pending`` is true we have already
                    # requested a keyframe and are waiting for the IDR that
                    # will seed the rebuilt decoder. Don't feed the existing
                    # transcoder any more deltas while we wait.
                    if self._transcoder is not None and not self._refresh_pending:
                        # Pre-decode RPS check: parse the first slice NAL's
                        # short-term RPS. If a used-by-curr reference points to
                        # a POC we never decoded, request a keyframe now and
                        # skip feeding this AU. ``check_slice`` also stamps the
                        # POC parser state regardless of outcome.
                        slice_nal = next(
                            (n for n in current_au if is_slice_nal((n[0] >> 1) & 0x3F)),
                            None,
                        )
                        missing: set[int] = set()
                        if slice_nal is not None:
                            missing = self._rps_tracker.check_slice(slice_nal)
                        if missing and not au_is_key:
                            logger.debug(
                                "rps: P-slice references missing POCs %s -- pre-decode PLI",
                                sorted(missing),
                            )
                            self._on_decode_error()
                            if self._refresh_pending:
                                current_au = []
                                au_is_key = False
                                au_corrupt = False
                                rtp_packets += 1
                                continue
                        annexb = b"".join(b"\x00\x00\x00\x01" + nal for nal in current_au)
                        self._transcoder.feed(annexb)
                        feed_count += 1
                        # Add this slice's POC to the DPB shadow so
                        # later slices can reference it. For IDRs
                        # this stamps POC=0; for P-slices it stamps
                        # the derived POC from check_slice above.
                        self._rps_tracker.commit_decoded()
                        # Sticky-keyframe IDR observation: half of the
                        # two-condition clear (the other half is a
                        # clean decoded frame, handled in
                        # ``_broadcast_frame``). Also stamps the
                        # post-IDR grace window so the in-flight
                        # pre-IDR P-frames in VT's queue don't
                        # re-trigger ``_on_decode_error``.
                        if au_is_key:
                            self._idr_observed_at = loop.time()
                            if self._keyframe_required:
                                self._idr_observed = True
                current_au = []
                au_is_key = False
                au_corrupt = False

            rtp_packets += 1
            now = loop.time()
            if now - last_stat_t >= 2.0:
                frames_now = self._frames_emitted
                logger.debug(
                    "ingress stats: rtp=%d feeds=%d frames=%d (Δframes=%d / %.1fs)",
                    rtp_packets,
                    feed_count,
                    frames_now,
                    frames_now - frame_count_at_last_log,
                    now - last_stat_t,
                )
                rtp_packets = 0
                feed_count = 0
                frame_count_at_last_log = frames_now
                last_stat_t = now

    # ----- RTCP PLI + decoder refresh ---------------------------------------
    def _build_rtcp_pli(self) -> bytes:
        """RTCP Picture Loss Indication (RFC 4585 §6.3.1). 12 bytes.

        Asks the device's encoder to emit a fresh IDR. The PLI is what
        gets us out from under stale long-term-reference state -- we
        send it, the device sends a new IDR, we rebuild the decoder
        from that IDR."""
        return struct.pack(
            "!BBHII",
            0x81,  # V=2, P=0, FMT=1
            0xCE,  # PT=206 (PSFB)
            2,  # length = 2 (3 words after this header)
            self._local_ssrc & 0xFFFFFFFF,
            self._remote_ssrc & 0xFFFFFFFF,
        )

    async def _send_rtcp_pli(self) -> None:
        transport = self._active_transport
        if transport is None or self._rtcp_dest is None:
            return
        if not (self._local_ssrc and self._remote_ssrc):
            return
        try:
            await transport.sendto(self._build_rtcp_pli(), *self._rtcp_dest)
            logger.debug("sent RTCP PLI (requested fresh keyframe)")
        except OSError as exc:
            logger.debug("PLI send failed (%s)", exc)

    def _build_rtcp_rr(self) -> bytes:
        """RTCP Receiver Report (RFC 3550 §6.4.2) + SDES, byte-compatible with
        screen_stream.py. The device's video streamConfig has RTCPTimeoutEnabled
        with a ~25 s interval; PLIs alone don't reset it -- without periodic RRs
        the device reaps the video session and stops emitting AUs, which froze
        serve-vnc permanently (it has no stream-restart). One RR/s keeps it alive.
        """
        rr = struct.pack(
            "!BBHII BBBB IIII",
            0x81,  # V=2, RC=1
            0xC9,  # PT=201 (RR)
            7,
            self._local_ssrc & 0xFFFFFFFF,
            self._remote_ssrc & 0xFFFFFFFF,
            0,  # fraction lost
            0,
            0,
            0,  # 3-byte cumulative loss
            self._rtp_highest_seq & 0xFFFFFFFF,
            0,  # interarrival jitter
            0,  # last SR
            0,  # delay since last SR
        )
        return rr + self._build_rtcp_sdes(self._local_ssrc)

    @staticmethod
    def _build_rtcp_sdes(ssrc: int) -> bytes:
        """Minimal SDES with an empty CNAME (matches Xcode's compound RR+SDES)."""
        return struct.pack("!BBHI BBBB", 0x81, 0xCA, 2, ssrc & 0xFFFFFFFF, 0x01, 0x00, 0x00, 0x00)

    async def _rtcp_send_loop(self, transport: UdpMediaTransport) -> None:
        """Send a video RR every second to keep the device's video session
        alive (see _build_rtcp_rr). Not gated on packets-received: send from the
        start so a brief no-frame window at bring-up can't let the session lapse.

        A send error (``OSError``) ends the task with that exception, which
        :meth:`wait_stopped` reports."""
        while True:
            try:
                await asyncio.sleep(1.0)
            except asyncio.CancelledError:
                return
            if self._rtcp_dest is None or not (self._local_ssrc and self._remote_ssrc):
                continue
            try:
                await transport.sendto(self._build_rtcp_rr(), *self._rtcp_dest)
            except OSError as exc:
                logger.debug("video RTCP RR send failed (%s); socket may be torn down", exc)
                raise

    def _fire_decoder_refresh(self, now: float, *, reason: str) -> None:
        """Arm the decoder-rebuild + PLI sequence. The recv loop will
        close the transcoder when the IDR arrives."""
        if self._refresh_pending:
            return  # already armed; let the IDR land first
        self._refresh_pending = True
        pli_task = asyncio.create_task(self._send_rtcp_pli())
        self._pli_tasks.add(pli_task)
        pli_task.add_done_callback(self._pli_tasks.discard)
        self._last_refresh_t = now
        window_bps = sum(s for _, s in self._au_byte_window)
        logger.debug(
            "decoder-refresh (%s): %d client(s), %d B/s window",
            reason,
            self._consumer_count(),
            window_bps,
        )

    async def _decoder_refresh_loop(self) -> None:
        """Watch the RTP byte rate as a motion signal and periodically request
        a keyframe + rebuild the transcoder: during sustained motion, once
        after motion settles, and as an idle heartbeat. Mirrors
        screen_stream's _decoder_refresh_loop.

        Triggers (in priority order):
          1. ``active``    -- motion ongoing, and ``active_interval`` elapsed
                              since the last refresh.
          2. ``settled``   -- motion just ended and ``settle_delay`` passed.
          3. ``heartbeat`` -- idle safety-net refresh.

        ``min_interval`` caps back-to-back refreshes (each costs a ~30 ms gap
        while the new decoder waits for its first IDR).
        """
        loop = asyncio.get_running_loop()
        motion_threshold_bps = 100_000  # 100 KB/s = motion
        active_interval = 1.5  # fire this often during sustained motion
        settle_delay = 0.4  # wait after motion ends
        heartbeat = 10.0  # max between refreshes when idle
        min_interval = 0.8  # cap back-to-back refreshes
        # Sticky-keyframe re-arm cadence + cap: while ``_keyframe_required``
        # and the encoder hasn't answered with a usable IDR, re-emit the
        # keyframe request every ``re_arm_interval`` (encoder IDR generate+
        # transmit settles in ~100-300 ms; 1.0 s leaves headroom). After
        # ``re_arm_cap`` attempts, give up and clear the flag so we stop
        # flooding; the motion/heartbeat triggers still catch the next
        # natural refresh window.
        re_arm_interval = 1.0
        re_arm_cap = 8
        while True:
            try:
                await asyncio.sleep(0.1)
            except asyncio.CancelledError:
                return
            if not self._consumer_count():
                continue
            if self._transcoder is None or self._rtcp_dest is None:
                continue
            now = loop.time()
            # Sticky re-arm: highest priority -- if the decoder is
            # still in keyframe_required after a previous refresh
            # cleared `_refresh_pending`, keep firing PLI on the
            # re-arm cadence until either a clean post-IDR frame
            # arrives (clears keyframe_required in `_broadcast_frame`)
            # or we hit the cap.
            if (
                self._keyframe_required
                and not self._refresh_pending
                and (now - self._last_refresh_t) >= re_arm_interval
            ):
                self._fir_attempts += 1
                if self._fir_attempts >= re_arm_cap:
                    if not self._cap_warned:
                        logger.warning(
                            "decoder did not recover after %d keyframe requests; "
                            "giving up sticky recovery (motion/heartbeat "
                            "refresh remains armed)",
                            self._fir_attempts,
                        )
                        self._cap_warned = True
                    self._keyframe_required = False
                    self._idr_observed = False
                    self._fir_attempts = 0
                else:
                    self._fire_decoder_refresh(now, reason="re-arm")
                    continue
            while self._au_byte_window and self._au_byte_window[0][0] < now - 1.0:
                self._au_byte_window.popleft()
            window_bytes = sum(s for _, s in self._au_byte_window)
            currently_active = window_bytes >= motion_threshold_bps
            if self._motion_active and not currently_active:
                self._motion_ended_t = now
            self._motion_active = currently_active
            since_refresh = now - self._last_refresh_t
            if since_refresh < min_interval:
                continue
            active = currently_active and since_refresh >= active_interval
            settled = self._motion_ended_t > self._last_refresh_t and (now - self._motion_ended_t) >= settle_delay
            heartbeat_due = since_refresh >= heartbeat
            if not (active or settled or heartbeat_due):
                continue
            reason = "active" if active else ("settled" if settled else "heartbeat")
            self._fire_decoder_refresh(now, reason=reason)

    # ----- Audio: AAC-ELD recv + decode + play ------------------------------
    async def _audio_udp_recv(self, transport: UdpMediaTransport) -> None:
        """Receive RTP audio packets, strip the RTP header, decode the
        AAC-ELD AU via AudioToolbox, and feed the PCM to the local
        AudioQueue. Mirrors :meth:`screen_stream.ScreenStreamServer._audio_udp_recv`
        but plays locally instead of broadcasting."""
        loop = asyncio.get_running_loop()
        consecutive_errors = 0
        _ERR_RECREATE_THRESHOLD = 5
        pkt_count = 0
        last_stat_t = loop.time()
        while True:
            try:
                data = await transport.recv()
            except (OSError, asyncio.CancelledError):
                return
            if len(data) < 12:
                continue
            pt = data[1] & 0x7F
            if 64 <= pt <= 95:  # RTCP -- ignore
                continue
            # Track sequence + receive count so our RR reports a sensible
            # extended-highest-seq field. Without this the device reaps
            # the audio session after 20 s (RTCPTimeoutInterval).
            self._audio_rtp_packets_received += 1
            seq = int.from_bytes(data[2:4], "big")
            cur_ext = self._audio_rtp_highest_seq
            cycles = (cur_ext >> 16) & 0xFFFF
            last_seq16 = cur_ext & 0xFFFF
            if seq < last_seq16 and (last_seq16 - seq) > 0x8000:
                cycles = (cycles + 1) & 0xFFFF
            new_ext = (cycles << 16) | seq
            if cur_ext == 0 or ((new_ext - cur_ext) & 0xFFFFFFFF) < 0x80000000:
                self._audio_rtp_highest_seq = new_ext
            cc = data[0] & 0x0F
            header_len = 12 + cc * 4
            if data[0] & 0x10:  # extension
                if header_len + 4 > len(data):
                    continue
                ext_len = int.from_bytes(data[header_len + 2 : header_len + 4], "big")
                header_len += 4 + ext_len * 4
            payload = data[header_len:]
            if not payload:
                continue
            decoder = self._audio_decoder
            player = self._audio_player
            if decoder is None or player is None:
                continue
            try:
                pcm = decoder.decode(payload)
                consecutive_errors = 0
            except Exception as exc:
                consecutive_errors += 1
                logger.debug("audio decode failed (%s) -- dropping", exc)
                if consecutive_errors >= _ERR_RECREATE_THRESHOLD:
                    logger.warning(
                        "audio decoder stuck after %d consecutive errors -- recreating",
                        consecutive_errors,
                    )
                    try:
                        self._audio_decoder = AACELDDecoder()
                        consecutive_errors = 0
                    except Exception:
                        logger.exception("audio decoder recreation failed")
                continue
            if pcm:
                player.play(pcm)
            pkt_count += 1
            now = loop.time()
            if now - last_stat_t >= 5.0:
                played, dropped, enq_err = player.stats()
                logger.debug(
                    "audio stats: rtp=%d, played=%d, dropped=%d, enq_err=%d",
                    pkt_count,
                    played,
                    dropped,
                    enq_err,
                )
                pkt_count = 0
                last_stat_t = now

    def _build_audio_rtcp_rr(self) -> bytes:
        """Audio-side counterpart of :meth:`_build_rtcp_pli` -- a minimal
        Receiver Report. The device's audio session has
        ``RTCPTimeoutInterval=20 s``; without these RRs the encoder
        stops emitting silently after ~20 s."""
        return struct.pack(
            "!BBHII BBBB IIII",
            0x81,  # V=2, P=0, RC=1
            0xC9,  # PT=201 (RR)
            7,
            self._audio_local_ssrc & 0xFFFFFFFF,
            self._audio_remote_ssrc & 0xFFFFFFFF,
            0,
            0,
            0,
            0,  # fraction lost / cumulative loss
            self._audio_rtp_highest_seq & 0xFFFFFFFF,
            0,
            0,
            0,  # jitter / lsr / dlsr
        )

    async def _audio_rtcp_send_loop(self, transport: UdpMediaTransport) -> None:
        sent = 0
        while True:
            try:
                await asyncio.sleep(1.0)
            except asyncio.CancelledError:
                return
            if self._audio_rtcp_dest is None or self._audio_rtp_packets_received == 0:
                continue
            try:
                await transport.sendto(self._build_audio_rtcp_rr(), *self._audio_rtcp_dest)
                sent += 1
                # One log line per 30 RRs (every 30 s) confirms the
                # session keepalive is firing -- useful when triaging
                # "audio stopped after a while" reports.
                if sent % 30 == 0:
                    logger.debug("audio RTCP RR: %d sent (highest_seq=0x%08x)", sent, self._audio_rtp_highest_seq)
            except OSError as exc:
                logger.debug("audio RTCP send failed (%s); socket may be torn down", exc)
                return

    # ----- HID --------------------------------------------------------------
    def _to_hid_coords(self, x: int, y: int) -> tuple[int, int]:
        """Clamp frame-pixel ``(x, y)`` and rescale it to the device's 0..65535 range."""
        hid_x = max(0, min(65535, int(x * 65535 / max(1, self._fb_width - 1))))
        hid_y = max(0, min(65535, int(y * 65535 / max(1, self._fb_height - 1))))
        return hid_x, hid_y

    async def send_touch(self, x: int, y: int, *, contact: bool) -> None:
        """Send a touchscreen contact (``contact=True``) or release at ``(x, y)`` in decoded-frame pixels.

        Out-of-frame coordinates are clamped. Ignored until the first frame is decoded, and
        outside a running session (before :meth:`start` brings the stream up, or after
        :meth:`stop`)."""
        if not self._accepting_input() or self._fb_width <= 0 or self._fb_height <= 0:
            return
        hid_x, hid_y = self._to_hid_coords(x, y)
        try:
            uhs = await self._ensure_hid()
            if uhs is None:
                return
            await uhs.send_touchscreen(
                TOUCHSCREEN_STATE_CONTACT if contact else TOUCHSCREEN_STATE_RELEASE,
                hid_x,
                hid_y,
                service_id=DIGITIZER_SURFACE_MAIN_TOUCHSCREEN,
            )
        except Exception:
            logger.exception("HID send failed")

    async def send_button(self, name: str, *, down: bool) -> None:
        """Press (``down=True``) or release a button from :data:`HARDWARE_BUTTONS`; ``KeyError`` if unknown.

        Ignored outside a running session (before :meth:`start` brings the stream up, or after
        :meth:`stop`)."""
        page, code = HARDWARE_BUTTONS[name]
        await self._send_hid_button(page, code, down=down)

    def _accepting_input(self) -> bool:
        """Input reaches the device only while the stream is up and ``stop()`` has not begun."""
        return self._started and not self._stopped

    async def _ensure_hid(self) -> Optional[UniversalHIDServiceService]:
        """The touchscreen/keyboard HID connection, opened on first use; ``None`` outside a running session."""
        async with self._hid_lock:
            if self._uhs is None and self._accepting_input():
                uhs = UniversalHIDServiceService(self._rsd)
                await uhs.connect()
                if self._stopped:
                    # stop() ran while we connected and its teardown already closed HID.
                    await _await_bounded(uhs.close(), STOP_STEP_TIMEOUT, "closing a late HID connection")
                    return None
                self._uhs = uhs
            return self._uhs

    async def _ensure_indigo(self) -> Optional[IndigoHIDService]:
        """The hardware-button HID connection, opened on first use; ``None`` outside a running session."""
        async with self._hid_lock:
            if self._indigo is None and self._accepting_input():
                ihs = IndigoHIDService(self._rsd)
                await ihs.connect()
                if self._stopped:
                    await _await_bounded(ihs.close(), STOP_STEP_TIMEOUT, "closing a late HID connection")
                    return None
                self._indigo = ihs
            return self._indigo

    async def _stop_hid(self) -> None:
        if self._uhs is not None:
            with contextlib.suppress(Exception):
                await self._uhs.close()
            self._uhs = None
        if self._indigo is not None:
            with contextlib.suppress(Exception):
                await self._indigo.close()
            self._indigo = None

    async def _ensure_keyboard(self) -> bool:
        """Register the virtual keyboard on first use; ``False`` outside a running session."""
        uhs = await self._ensure_hid()
        if uhs is None:
            return False
        if self._kb_service_id is None:
            async with self._hid_lock:
                if self._kb_service_id is None:
                    self._kb_service_id = await uhs.create_keyboard_service()
        return True

    async def _send_hid_button(self, page: int, code: int, *, down: bool) -> None:
        if not self._accepting_input():
            return
        state = HID_BUTTON_STATE_DOWN if down else HID_BUTTON_STATE_UP
        try:
            indigo = await self._ensure_indigo()
            if indigo is None:
                return
            await indigo.send_button(page, code, state)
        except Exception:
            logger.exception("HID button send failed (page=0x%02X code=0x%02X)", page, code)

    # ----- top-level orchestration ------------------------------------------
    async def start(self) -> None:
        """Start the device video (and optional audio) stream and the background tasks.

        Returns once the stream is negotiated; use :meth:`wait_first_frame` for the first frame.

        The session is single-use: calling ``start()`` again, or after :meth:`stop`, raises
        ``RuntimeError``. If ``start()`` fails, everything it created is closed before the
        exception propagates, so no :meth:`stop` is needed. A concurrent :meth:`stop` interrupts
        an in-flight ``start()`` (it cancels the device call, since a hung one can never notice a
        flag): ``start()`` unwinds and raises :class:`~pymobiledevice3.exceptions.ScreenMirrorStopped`,
        and the caller's task is not left cancelled. Cancelling the caller's task itself still
        propagates ``asyncio.CancelledError``.
        """
        if self._start_called or self._stopped:
            raise RuntimeError("ScreenMirrorSession is single-use: start() was already called or the session stopped")
        self._start_called = True
        self._start_task = asyncio.current_task()
        self._start_exited.clear()
        try:
            await self._start_streams()
        except BaseException as e:
            if not self._stopped:
                # A genuine failure, not stop() cancelling us (stop() tears down itself): unwind
                # here. Release the device session only if the stream came up -- a rejected
                # start_video_stream left nothing to release on the device.
                self._stopped = True
                try:
                    await self._teardown(release_device=self._started)
                finally:
                    self._teardown_done.set()
                    self._stopped_event.set()
            if self._cancelled_by_stop:
                # stop() cancelled us (whatever the interrupted device call turned that into).
                if self._withdraw_stop_cancellation():
                    raise ScreenMirrorStopped("stop() interrupted start()") from None
                # The caller's task was cancelled as well: that cancellation wins.
                if not isinstance(e, asyncio.CancelledError):
                    raise asyncio.CancelledError() from e
                raise
            if isinstance(e, CoreDeviceError) and is_media_in_use_error(e):
                # Camera/microphone in use: there is nothing to serve. Fail fast
                # with a clean, actionable error (no traceback).
                raise CoreDeviceError(MEDIA_IN_USE_MESSAGE, code=e.code) from None
            raise
        finally:
            self._start_task = None
            self._start_exited.set()

    def _withdraw_stop_cancellation(self) -> bool:
        """Undo the ``cancel()`` that :meth:`stop` sent the task running :meth:`start`.

        Returns ``True`` when no other cancellation of that task is pending, i.e. the
        cancellation came from ``stop()`` alone and may be reported as ScreenMirrorStopped.
        Before Python 3.11 a task keeps no cancellation count, so the flag alone decides."""
        self._cancelled_by_stop = False
        if sys.version_info >= (3, 11):
            task = asyncio.current_task()
            if task is not None:
                return task.uncancel() == 0
        return True

    async def _start_streams(self) -> None:
        """The body of :meth:`start`. Everything it creates goes on ``self`` as soon as it
        exists, so :meth:`_teardown` can close it whenever this is interrupted."""
        # 1) Start device-side video stream + open the RTP receiver on the right transport:
        # the pytcp stack over the userspace tunnel (a host kernel socket is unreachable from
        # the device), a host kernel socket otherwise. Generate a single client_session_id up
        # front so the audio stream (started below) gets paired with the video session on the
        # device -- this is what Xcode's Mirror does.
        shared_session_id = uuid.uuid4()
        svc = DisplayService(self._rsd)
        self._svc = svc
        await svc.connect()
        transport, receiver_ip = open_media_receiver(svc, (8 * 1024 * 1024, 4 * 1024 * 1024))
        self._transport = transport
        self._stream_requested = True
        answer = await svc.start_video_stream(
            receiver_ip=receiver_ip,
            receiver_port=transport.port,
            sender_ip=self._sender_ip,
            display_id=self._display_id,
            client_session_id=shared_session_id,
            allow_rtcp_fb=self._allow_rtcp_fb,
            ltrp_enabled=self._ltrp_enabled,
        )
        self._started = True
        cfg = answer["connection"].get("streamConfig", {})
        logger.info(
            "video stream up: %dx%d HEVC, sender_port=%s",
            int(cfg.get("CustomWidth", 0)),
            int(cfg.get("CustomHeight", 0)),
            cfg.get("SourcePort"),
        )

        # RTCP feedback setup: ``LocalSSRC``/``RemoteSSRC`` in
        # streamConfig are from the device's perspective, so OURS is
        # ``RemoteSSRC`` and THEIRS is ``LocalSSRC`` (yes, really).
        # Without all three of (source_port, local_ssrc, remote_ssrc)
        # we can't send PLI and the decoder-refresh loop will skip.
        source_port = int(cfg.get("SourcePort", 0))
        self._local_ssrc = int(cfg.get("RemoteSSRC", 0))
        self._remote_ssrc = int(cfg.get("LocalSSRC", 0))
        self._rtcp_dest = (self._sender_ip, source_port) if source_port else None
        self._active_transport = transport
        if self._rtcp_dest and self._local_ssrc and self._remote_ssrc:
            logger.info(
                "RTCP feedback enabled: dest=%s, ours=%d, theirs=%d",
                self._rtcp_dest,
                self._local_ssrc,
                self._remote_ssrc,
            )
        else:
            logger.warning(
                "RTCP feedback disabled (missing fields: SourcePort=%s LocalSSRC=%s RemoteSSRC=%s)"
                " -- keyframe requests / decoder-refresh will be a no-op",
                source_port,
                self._local_ssrc,
                self._remote_ssrc,
            )

        # 3) Audio stream (optional). RFB has no audio; we play the
        # decoded PCM through the host's speakers via AudioQueue. The
        # device's audio session shares ``client_session_id`` with the
        # video session so they're paired on the device-side media
        # manager (same as Xcode's Mirror). Without RR keepalives the
        # device reaps the audio session after ~20 s.
        if self._audio_enabled:
            try:
                audio_svc = DisplayService(self._rsd)
                self._audio_svc = audio_svc
                await audio_svc.connect()
                audio_transport, audio_receiver_ip = open_media_receiver(audio_svc, (4 * 1024 * 1024, 1 * 1024 * 1024))
                self._audio_transport = audio_transport
                audio_answer = await audio_svc.start_audio_stream(
                    receiver_ip=audio_receiver_ip,
                    receiver_port=audio_transport.port,
                    sender_ip=self._sender_ip,
                    client_session_id=shared_session_id,
                )
                audio_sid_raw = audio_answer["connection"]["options"]["avcMediaStreamOptionClientSessionID"]["uuid"]
                audio_sid = audio_sid_raw if isinstance(audio_sid_raw, uuid.UUID) else uuid.UUID(audio_sid_raw)
                audio_cfg = audio_answer["connection"].get("streamConfig", {})
                a_source_port = int(audio_cfg.get("SourcePort", 0))
                self._audio_local_ssrc = int(audio_cfg.get("RemoteSSRC", 0))
                self._audio_remote_ssrc = int(audio_cfg.get("LocalSSRC", 0))
                self._audio_rtcp_dest = (self._sender_ip, a_source_port) if a_source_port else None
                self._audio_session_id = audio_sid
                self._audio_decoder = AACELDDecoder()
                self._audio_player = AudioQueuePlayer()
                logger.info(
                    "audio stream up: PT=%s mode=%s sender_port=%s",
                    audio_cfg.get("RxPayloadType"),
                    audio_cfg.get("AudioStreamMode"),
                    a_source_port,
                )
            except Exception:
                logger.exception("audio startup failed -- continuing without audio")
                self._audio_enabled = False
                # Tear down anything we managed to create before the error.
                if self._audio_player is not None:
                    with contextlib.suppress(Exception):
                        self._audio_player.close()
                    self._audio_player = None
                self._audio_decoder = None
                if self._audio_transport is not None:
                    with contextlib.suppress(Exception):
                        self._audio_transport.close()
                    self._audio_transport = None
                if self._audio_svc is not None:
                    with contextlib.suppress(Exception):
                        await asyncio.wait_for(self._audio_svc.close(), STOP_STEP_TIMEOUT)
                    self._audio_svc = None
                self._audio_session_id = None

        # 4) Background tasks. A device call above may have swallowed stop()'s cancellation:
        # bail out rather than start tasks that the (already running) teardown would miss.
        if self._stopped:
            raise ScreenMirrorStopped("stop() interrupted start()")
        loop = asyncio.get_running_loop()
        self._loop = loop
        self._feed_task = asyncio.create_task(self._udp_recv_and_pipe(transport))
        self._feed_task.add_done_callback(self._on_stream_task_done)
        self._refresh_task = asyncio.create_task(self._decoder_refresh_loop())
        # Video RR keepalive -- without this the device reaps the video session
        # after ~25 s (RTCPTimeoutEnabled) and serve-vnc freezes with no recovery.
        self._rtcp_task = asyncio.create_task(self._rtcp_send_loop(transport))
        self._rtcp_task.add_done_callback(self._on_stream_task_done)
        if self._audio_enabled and self._audio_transport is not None:
            self._audio_recv_task = asyncio.create_task(self._audio_udp_recv(self._audio_transport))
            if self._audio_rtcp_dest and self._audio_local_ssrc and self._audio_remote_ssrc:
                self._audio_rtcp_task = asyncio.create_task(self._audio_rtcp_send_loop(self._audio_transport))
            else:
                logger.warning(
                    "audio RTCP disabled (missing fields in streamConfig) -- session will be reaped after ~20 s"
                )

    async def stop(self) -> None:
        """Tear down everything :meth:`start` created. Idempotent; a no-op before ``start()``.

        Bounded even when the device is gone and every device call hangs: the HID and stream
        connections close concurrently within ``STOP_STEP_TIMEOUT``, then releasing the
        device-side media session gets ``STOP_ALL_STREAMS_TIMEOUT``. Called while :meth:`start`
        is still in flight, ``stop()`` interrupts that ``start()`` (which then raises
        :class:`~pymobiledevice3.exceptions.ScreenMirrorStopped`) and tears down what it had
        created. There is one teardown,
        whoever runs it (the first ``stop()`` or a failed ``start()``): every ``stop()`` call
        returns only once it has finished. Once ``stop()`` returns, :meth:`wait_stopped` does
        too. The session cannot be started again.
        """
        if self._stopped:
            await self._teardown_done.wait()
            return
        self._stopped = True
        try:
            if not self._start_called:
                return
            start_task = self._start_task
            if start_task is not None and start_task is not asyncio.current_task():
                logger.debug("shutdown: cancelling the in-flight start()")
                self._cancelled_by_stop = True
                start_task.cancel()
            await self._teardown(release_device=self._stream_requested, start_exited=self._start_exited)
        finally:
            self._teardown_done.set()
            self._stopped_event.set()

    async def _drain_pli_tasks(self) -> None:
        if self._pli_tasks:
            await asyncio.wait(list(self._pli_tasks), timeout=1.0)

    async def _teardown(self, *, release_device: bool, start_exited: Optional[asyncio.Event] = None) -> None:
        """Close whatever :meth:`start` created, with every device call bounded (see :meth:`stop`).

        ``start_exited`` is awaited within the same step as the connection closes, so waiting
        for a cancelled in-flight ``start()`` to unwind costs no extra time."""
        try:
            logger.debug("shutdown: stopping background tasks")
            tasks = [
                t
                for t in (
                    self._refresh_task,
                    self._rtcp_task,
                    self._feed_task,
                    self._audio_recv_task,
                    self._audio_rtcp_task,
                )
                if t is not None
            ]
            for task in tasks:
                task.cancel()
            for task in tasks:
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
            logger.debug("shutdown: stopping VT transcoder")
            if self._transcoder is not None:
                with contextlib.suppress(Exception):
                    self._transcoder.close()
                self._transcoder = None
            if self._audio_player is not None:
                logger.debug("shutdown: stopping AudioQueue player")
                with contextlib.suppress(Exception):
                    self._audio_player.close()
                self._audio_player = None

            # One bounded step: HID, the stream connections, the in-flight PLIs (they still use
            # the transport) and a cancelled start() all wind down concurrently.
            logger.debug("shutdown: closing HID and device stream connections")
            steps: list[Awaitable[Any]] = [self._stop_hid(), self._drain_pli_tasks()]
            if start_exited is not None:
                steps.append(start_exited.wait())
            for svc in (self._audio_svc, self._svc):
                if svc is not None:
                    steps.append(svc.close())
            await _await_bounded(
                asyncio.gather(*steps, return_exceptions=True), STOP_STEP_TIMEOUT, "closing device connections"
            )
        finally:
            # Synchronous, so it runs even when this teardown is cancelled: the sockets and the
            # in-flight PLIs never outlive it.
            self._uhs = None
            self._indigo = None
            self._svc = None
            self._audio_svc = None
            for task in list(self._pli_tasks):
                task.cancel()
            for transport in (self._transport, self._audio_transport):
                if transport is not None:
                    with contextlib.suppress(Exception):
                        transport.close()
            self._transport = None
            self._audio_transport = None

        # Release the device-side session, so the device clears its
        # remote-observation state and frees the camera/microphone (see
        # DisplayService.stop_all_streams).
        if release_device:
            logger.debug("shutdown: releasing device media session")
            await _await_bounded(
                DisplayService.stop_all_streams(self._rsd),
                STOP_ALL_STREAMS_TIMEOUT,
                "releasing the device media session",
            )
