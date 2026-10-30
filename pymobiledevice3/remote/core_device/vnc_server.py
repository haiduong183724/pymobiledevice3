"""
RFB 3.8 (VNC) server for the device's screen.

Open via macOS Finder ``Cmd+K`` → ``vnc://127.0.0.1:5901`` (or any
VNC client). No browser, no WebCodecs, no canvas state to corrupt --
each framebuffer update is a fresh BGRA frame straight from the
VideoToolbox HEVC decoder.

Pipeline: RTP/HEVC -> VTDecompressionSession (BGRA output) -> Raw VNC.

No JPEG round-trip on this path. The previous design encoded BGRA ->
JPEG -> decoded JPEG -> BGRA again before sending; macOS Screen
Sharing's client doesn't advertise Tight, so the JPEG was always
thrown away. Skipping it removes the second lossy step (q=0.7 JPEG)
and the encode+decode CPU work.

Only the Raw encoding is offered. Raw is mandatory per RFC 6143 §7.7.1
-- every VNC client supports it, including macOS Screen Sharing and
every third-party viewer that doesn't speak Apple-private encodings.

Protocol reference: RFC 6143 (RFB 3.8).
"""

import asyncio
import contextlib
import hmac
import logging
import os
import select
import socket
import struct
import zlib
from typing import Optional

from cryptography.hazmat.decrepit.ciphers.algorithms import TripleDES
from cryptography.hazmat.primitives.ciphers import Cipher, modes

from pymobiledevice3.remote.core_device import rfb_clipboard
from pymobiledevice3.remote.core_device.hid_service import (
    ASCII_TO_HID,
    KEY_BACKSPACE,
    KEY_CAPS_LOCK,
    KEY_DOWN,
    KEY_ENTER,
    KEY_ESC,
    KEY_F1,
    KEY_LEFT,
    KEY_LEFT_ALT,
    KEY_LEFT_CTRL,
    KEY_LEFT_GUI,
    KEY_LEFT_SHIFT,
    KEY_RIGHT,
    KEY_RIGHT_ALT,
    KEY_RIGHT_CTRL,
    KEY_RIGHT_GUI,
    KEY_RIGHT_SHIFT,
    KEY_TAB,
    KEY_UP,
)
from pymobiledevice3.remote.core_device.pasteboard_service import PasteboardContent, PasteboardMonitor
from pymobiledevice3.remote.core_device.screen_mirror import (
    _BTN_HOME,
    _BTN_LOCK,
    _BTN_MUTE,
    _BTN_SIRI,
    _BTN_VOL_DOWN,
    _BTN_VOL_UP,
    ScreenMirrorSession,
)
from pymobiledevice3.remote.remote_service_discovery import RemoteServiceDiscoveryService

logger = logging.getLogger(__name__)


# RFB encoding identifiers
_ENC_RAW = 0
_ENC_COPY_RECT = 1
# pseudo-encodings (negative int32, but client lists them as positive in
# SetEncodings; we just match what the client advertises)
_ENC_CURSOR = -239
_ENC_DESKTOP_SIZE = -223
_ENC_LAST_RECT = -224

_SERVER_NAME = b"iPhone screen (pymobiledevice3)"


# RFB mouse-button bit (zero-indexed) -> iOS button.
# Left button (bit 0) is the touch-pointer, handled separately.
_MOUSE_BUTTON_TO_HID: dict[int, tuple[int, int]] = {
    1: _BTN_LOCK,  # middle click
    2: _BTN_HOME,  # right click
}

# Modifier keysyms. macOS Screen Sharing sends each modifier as its
# own KeyEvent down/up around the letter, so we track state per
# client and gate the hotkey lookup on Ctrl being held.
_KEYSYM_CTRL = frozenset({0xFFE3, 0xFFE4})  # Control_L / Control_R
_KEYSYM_SHIFT = frozenset({0xFFE1, 0xFFE2})
_KEYSYM_ALT = frozenset({0xFFE9, 0xFFEA})  # Alt_L (Option) / Alt_R
_KEYSYM_CMD = frozenset({0xFFE7, 0xFFE8, 0xFFEB, 0xFFEC})  # Meta_L/R or Super_L/R

# X11 modifier-keysym -> HID modifier usage. Both sides of the keyboard
# are kept so a client distinguishing L/R shift still works (we just
# treat them as separate bits in the bitmap).
_KEYSYM_TO_HID_MODIFIER: dict[int, int] = {
    0xFFE1: KEY_LEFT_SHIFT,
    0xFFE2: KEY_RIGHT_SHIFT,
    0xFFE3: KEY_LEFT_CTRL,
    0xFFE4: KEY_RIGHT_CTRL,
    0xFFE7: KEY_LEFT_GUI,
    0xFFE8: KEY_RIGHT_GUI,  # Meta_L / Meta_R
    0xFFE9: KEY_LEFT_ALT,
    0xFFEA: KEY_RIGHT_ALT,
    0xFFEB: KEY_LEFT_GUI,
    0xFFEC: KEY_RIGHT_GUI,  # Super_L / Super_R
}

# X11 special keysyms (non-printable) -> HID usage.
_KEYSYM_TO_HID_SPECIAL: dict[int, int] = {
    0xFF08: KEY_BACKSPACE,
    0xFF09: KEY_TAB,
    0xFF0D: KEY_ENTER,
    0xFF8D: KEY_ENTER,  # Return / KP_Enter
    0xFF1B: KEY_ESC,
    0xFFE5: KEY_CAPS_LOCK,
    0xFF50: 0x4A,  # Home
    0xFF51: KEY_LEFT,
    0xFF52: KEY_UP,
    0xFF53: KEY_RIGHT,
    0xFF54: KEY_DOWN,
    0xFF55: 0x4B,  # Page_Up
    0xFF56: 0x4E,  # Page_Down
    0xFF57: 0x4D,  # End
    0xFF63: 0x49,  # Insert
    0xFFFF: 0x4C,  # Delete (forward)
    # F1..F12 are sequential keysyms 0xFFBE..0xFFC9 and sequential HID
    # usages KEY_F1..KEY_F12.
    **{0xFFBE + i: KEY_F1 + i for i in range(12)},
}


# Ctrl + keysym -> iOS button. Ctrl is the prefix because:
# - No Fn required: every Mac keyboard has a real Ctrl key.
# - Cmd never reaches us: Screen Sharing.app intercepts Cmd+anything
#   for its own window shortcuts.
# - Option+letter emits special chars on Mac (Opt+s -> ß etc.) which
#   Screen Sharing forwards as those chars, not the base letter.
# - Reserves plain letters / symbols for the upcoming text-input HID
#   path, so hotkeys won't collide with typing once we add it.
# Both ASCII cases are mapped, so Caps Lock or held-Shift don't break
# a hotkey (e.g. Ctrl+] vs Ctrl+}).
_CTRL_COMBO_TO_HID: dict[int, tuple[int, int]] = {
    ord("h"): _BTN_HOME,
    ord("H"): _BTN_HOME,
    ord("l"): _BTN_LOCK,
    ord("L"): _BTN_LOCK,
    ord("["): _BTN_VOL_DOWN,
    ord("{"): _BTN_VOL_DOWN,
    ord("]"): _BTN_VOL_UP,
    ord("}"): _BTN_VOL_UP,
    ord("\\"): _BTN_MUTE,
    ord("|"): _BTN_MUTE,
    ord("s"): _BTN_SIRI,
    ord("S"): _BTN_SIRI,
}


class _VncClient:
    """Per-connection state."""

    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self.reader = reader
        self.writer = writer
        self.encodings: list[int] = []
        self.wants_update = asyncio.Event()
        # Identity-keyed dedupe: the broadcaster swaps in a new ``bytes``
        # object per frame, so reference equality is enough.
        self.last_sent_frame: Optional[bytes] = None
        # Pointer button state for click->drag->release synthesis.
        self.pressed = False
        self.last_x = 0
        self.last_y = 0
        # Previous RFB button mask, used to diff non-left buttons
        # (right/middle) on each PointerEvent so we can fire the
        # mapped iOS hardware button on the down->up edge.
        self.last_button_mask = 0
        # Modifier state -- Ctrl is the only one consulted for hotkey
        # gating today; the rest are tracked for symmetry / future use.
        self.mod_ctrl = False
        self.mod_shift = False
        self.mod_alt = False
        self.mod_cmd = False
        # keysym -> (page, code) for hotkeys whose key-down fired a
        # button-down. We need this so that on key-up we send the
        # matching button-up regardless of whether the user released
        # Ctrl first.
        self.active_combos: dict[int, tuple[int, int]] = {}
        # Currently-held HID keyboard usage codes. Every key event
        # rewrites this set and re-emits the full bitmap report.
        self.pressed_keys: set[int] = set()
        # keysym -> set of usage codes the key-down emitted. Tracked so
        # key-up can release the same usages, even if the host's
        # interpretation of the keysym (e.g. shifted vs unshifted) has
        # drifted between the down and the up.
        self.active_typing: dict[int, tuple[int, ...]] = {}
        # Small server->client messages (clipboard) waiting for the send loop,
        # which owns the send direction of the socket.
        self.outbox: list[bytes] = []
        # The client advertised the Extended Clipboard pseudo-encoding (UTF-8).
        self.extended_clipboard = False


def vnc_auth_response(password: str, challenge: bytes) -> bytes:
    """What a client answers to a VNC Auth challenge for ``password``.

    RFB VNC Authentication: the password, truncated or NUL-padded to 8 bytes, is the DES key,
    with the bits of each key byte reversed (a quirk every VNC implementation carries); the 16
    byte challenge is encrypted with it in ECB mode. DES is only in ``cryptography`` as
    single-key 3DES these days, which is the same cipher.
    """
    key = bytes(int(f"{byte:08b}"[::-1], 2) for byte in password.encode("utf-8")[:8].ljust(8, b"\0"))
    encryptor = Cipher(TripleDES(key * 3), modes.ECB()).encryptor()
    return encryptor.update(challenge) + encryptor.finalize()


class VncStreamServer(ScreenMirrorSession):
    """RFB 3.8 server. Streams the device screen as Raw BGRA
    framebuffer updates."""

    def __init__(
        self,
        rsd: RemoteServiceDiscoveryService,
        *,
        bind: str = "127.0.0.1",
        port: int = 5901,
        display_id: int = 1,
        password: Optional[str] = None,
        audio: bool = False,
        decoder: str = "auto",
        allow_rtcp_fb: bool = False,
        ltrp_enabled: bool = False,
        share_clipboard: bool = False,
    ) -> None:
        super().__init__(
            rsd,
            display_id=display_id,
            audio=audio,
            decoder=decoder,
            allow_rtcp_fb=allow_rtcp_fb,
            ltrp_enabled=ltrp_enabled,
        )
        # --share-clipboard: device copies go out as ServerCutText, ClientCutText
        # lands on the device pasteboard.
        self._clipboard_monitor: Optional[PasteboardMonitor] = (
            PasteboardMonitor(rsd, self._on_device_clipboard) if share_clipboard else None
        )
        self._clipboard_text: Optional[str] = None
        self._bind = bind
        self._port = port
        self._password = password
        self._clients: set[_VncClient] = set()

    def _broadcast_frame(self, bgra: bytes) -> None:
        super()._broadcast_frame(bgra)
        for c in self._clients:
            c.wants_update.set()

    def _consumer_count(self) -> int:
        return len(self._clients)

    async def _handle_pointer(self, client: _VncClient, button_mask: int, x: int, y: int) -> None:
        """Translate an RFB PointerEvent into HID activity.

        Left button (bit 0)  -> touchscreen contact at (x, y).
        Right button (bit 2) -> iOS Home button.
        Middle button (bit 1) -> iOS Lock/Power button.

        Touch coords are rescaled from framebuffer pixels to the
        device's uint16 normalised range (0..65535)."""
        if self._fb_width <= 0 or self._fb_height <= 0:
            return

        # Diff non-left buttons for press/release edge detection.
        changed = button_mask ^ client.last_button_mask
        for bit, (page, code) in _MOUSE_BUTTON_TO_HID.items():
            mask = 1 << bit
            if changed & mask:
                await self._send_hid_button(page, code, down=bool(button_mask & mask))
        client.last_button_mask = button_mask

        hid_x, hid_y = self._to_hid_coords(x, y)
        pressed = bool(button_mask & 1)
        # Release only on the pressed -> released transition, at the last drag position.
        if pressed or client.pressed:
            await self.send_touch(x, y, contact=pressed)
        client.pressed = pressed
        client.last_x = hid_x
        client.last_y = hid_y

    async def _handle_key(self, client: _VncClient, down: bool, keysym: int) -> None:
        """Translate an RFB KeyEvent into HID activity.

        Three paths:

        - Ctrl+<key> combos in ``_CTRL_COMBO_TO_HID`` fire an Indigo hardware
          button (Home / Lock / Vol / Mute / Siri).
        - Modifier keysyms (Shift/Ctrl/Alt/GUI) toggle the matching HID
          modifier usages in the virtual-keyboard bitmap, and also update
          the legacy ``mod_*`` flags that the Ctrl-combo path consults.
        - Anything else -- printable ASCII via :data:`ASCII_TO_HID`, or
          named keys via ``_KEYSYM_TO_HID_SPECIAL`` -- toggles its usage in
          the virtual-keyboard bitmap and re-emits the full report.

        Key-up always releases the same usages the matching key-down
        emitted, even if the host's interpretation of the keysym drifted
        in between (e.g. Shift released before the letter).
        """
        # Track the legacy modifier-bool state so the Ctrl-combo path
        # below keeps working alongside the keyboard path.
        if keysym in _KEYSYM_CTRL:
            client.mod_ctrl = down
        elif keysym in _KEYSYM_SHIFT:
            client.mod_shift = down
        elif keysym in _KEYSYM_ALT:
            client.mod_alt = down
        elif keysym in _KEYSYM_CMD:
            client.mod_cmd = down

        # Ctrl-prefixed hardware-button hotkeys. Eat the event so the
        # letter doesn't also get typed on the device.
        if down and client.mod_ctrl and keysym in _CTRL_COMBO_TO_HID:
            mapping = _CTRL_COMBO_TO_HID[keysym]
            client.active_combos[keysym] = mapping
            await self._send_hid_button(mapping[0], mapping[1], down=True)
            return
        if not down and keysym in client.active_combos:
            active = client.active_combos.pop(keysym)
            await self._send_hid_button(active[0], active[1], down=False)
            return

        # Resolve the keysym to one or more HID usages.
        usages: tuple[int, ...] = ()
        if keysym in _KEYSYM_TO_HID_MODIFIER:
            usages = (_KEYSYM_TO_HID_MODIFIER[keysym],)
        elif keysym in _KEYSYM_TO_HID_SPECIAL:
            usages = (_KEYSYM_TO_HID_SPECIAL[keysym],)
        elif 0x20 <= keysym <= 0x7E:
            mapping = ASCII_TO_HID.get(chr(keysym))
            if mapping is not None:
                usage, needs_shift = mapping
                # Shifted character on key-down: synthesise Shift in the
                # bitmap if no client-tracked Shift is already held.
                if needs_shift and not (
                    KEY_LEFT_SHIFT in client.pressed_keys or KEY_RIGHT_SHIFT in client.pressed_keys
                ):
                    usages = (KEY_LEFT_SHIFT, usage)
                else:
                    usages = (usage,)
        if not usages:
            return

        if down:
            client.active_typing[keysym] = usages
            client.pressed_keys.update(usages)
        else:
            for u in client.active_typing.pop(keysym, usages):
                client.pressed_keys.discard(u)
        try:
            if not await self._ensure_keyboard():
                return  # the session is not running
            assert self._uhs is not None and self._kb_service_id is not None
            await self._uhs.send_keyboard(self._kb_service_id, client.pressed_keys)
        except Exception:
            logger.exception("HID keyboard send failed (keysym=0x%04X down=%s)", keysym, down)

    # ----- RFB protocol -----------------------------------------------------
    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        peer = writer.get_extra_info("peername")
        client = _VncClient(reader, writer)
        logger.info("VNC client connected: %s", peer)
        try:
            # Wait for the first frame so we have framebuffer dimensions.
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(self._ready.wait(), timeout=10.0)
            if not self._ready.is_set():
                logger.warning("VNC client %s: stream not ready, dropping", peer)
                return
            await self._handshake(client)
            self._clients.add(client)
            # Force a decoder rebuild for this client. The VT decoder
            # may have been accumulating stale LTRP state for the
            # entire time the server has been running before this
            # client connected; trigger the same refresh path that
            # motion-settled and heartbeat use so the very first
            # framebuffer update is decoded from a fresh IDR.
            if self._transcoder is not None and self._rtcp_dest is not None:
                loop = self._loop
                if loop is not None:
                    self._fire_decoder_refresh(loop.time(), reason="client-connect")
            await asyncio.gather(
                self._client_send_loop(client),
                self._client_recv_loop(client),
            )
        except (asyncio.IncompleteReadError, ConnectionResetError, BrokenPipeError, asyncio.CancelledError):
            pass
        except ConnectionError as e:
            # the handshake turned the client away (wrong password, unsupported security type)
            logger.warning("VNC client %s rejected: %s", peer, e)
        except Exception:
            logger.exception("VNC client crashed: %s", peer)
        finally:
            self._clients.discard(client)
            with contextlib.suppress(Exception):
                writer.close()
                await writer.wait_closed()
            logger.info("VNC client disconnected: %s", peer)

    async def _handshake(self, client: _VncClient) -> None:
        r = client.reader
        w = client.writer
        # 1. ProtocolVersion (12 bytes each way).
        w.write(b"RFB 003.008\n")
        await w.drain()
        logger.debug("handshake: sent server version RFB 003.008")
        client_version_bytes = await r.readexactly(12)
        client_version = client_version_bytes.decode("ascii", errors="replace").rstrip()
        logger.debug("handshake: client version %r", client_version)
        # Parse minor version. The protocol changes between 3.3 / 3.7 /
        # 3.8 across the security handshake:
        #
        #   3.3 -- server unilaterally picks ONE security type and sends
        #          it as a U32. No SecurityResult on None auth.
        #   3.7 -- server sends a list (count + types[]), client picks.
        #          No SecurityResult on None auth.
        #   3.8 -- server sends a list, client picks, server always
        #          sends SecurityResult.
        #
        # macOS Screen Sharing.app picks 3.3 (regardless of our offer)
        # because that's the protocol Apple Remote Desktop wraps. So
        # we MUST support 3.3 if we want to be reachable from Finder
        # Cmd+K -> vnc:// out of the box.
        minor = 8
        try:
            if client_version.startswith("RFB 003."):
                minor = int(client_version[8:11])
        except ValueError:
            minor = 8
        # Always VNC Auth (security type 2): macOS Screen Sharing's password
        # dialog refuses an empty entry even when the server advertises None
        # auth. With a password configured the response is checked; without
        # one, whatever the client sends is accepted (the server is
        # loopback-only by default, and the CLI says so).
        if minor < 7:
            # RFB 3.3: server unilaterally picks the security type.
            w.write(struct.pack(">I", 2))  # VNC Auth
            await w.drain()
            logger.debug("handshake: 3.3 path -- server picked VNC Auth")
        else:
            # RFB 3.7 / 3.8: send security list, client picks. Offer
            # only VNC Auth so we always end up in the same code path.
            w.write(b"\x01\x02")
            await w.drain()
            logger.debug("handshake: sent security types [VNC Auth=2]")
            chosen = (await r.readexactly(1))[0]
            logger.debug("handshake: client picked security=%d", chosen)
            if chosen != 2:
                msg = b"unsupported security type"
                w.write(struct.pack(">I", 1) + struct.pack(">I", len(msg)) + msg)
                await w.drain()
                raise ConnectionError(f"client picked unsupported security={chosen}")
        # VNC Auth challenge/response: 16 random bytes out, their DES
        # encryption under the password back (see vnc_auth_response).
        challenge = os.urandom(16)
        w.write(challenge)
        await w.drain()
        response = await r.readexactly(16)
        if self._password is not None and not hmac.compare_digest(
            response, vnc_auth_response(self._password, challenge)
        ):
            # SecurityResult=failed; 3.8 clients also get a reason string.
            reason = b"Authentication failed"
            w.write(struct.pack(">I", 1) + (struct.pack(">I", len(reason)) + reason if minor >= 8 else b""))
            await w.drain()
            raise ConnectionError("VNC Auth failed")
        logger.debug("handshake: VNC Auth accepted (%s)", "password" if self._password else "any password")
        # 3.x always sends SecurityResult AFTER VNC Auth, regardless of
        # whether None auth would have skipped it.
        w.write(b"\x00\x00\x00\x00")
        await w.drain()
        logger.debug("handshake: sent SecurityResult=OK")
        # ClientInit (shared flag — we don't care).
        shared = (await r.readexactly(1))[0]
        logger.debug("handshake: client shared=%d", shared)
        # 5. ServerInit: width, height, pixel format, name.
        # Pixel format = 32bpp little-endian BGRA. (For Tight-JPEG the
        # client doesn't need this to match the JPEG; for Raw we'd
        # need to emit BGRA bytes -- this server only emits Tight-JPEG
        # frames so the pixel format is mostly cosmetic but we still
        # advertise something sensible.)
        pixel_format = struct.pack(
            ">BBBB HHH BBB 3x",
            32,  # bits-per-pixel
            24,  # depth
            0,  # big-endian-flag
            1,  # true-colour-flag
            255,
            255,
            255,  # red/green/blue max
            16,
            8,
            0,  # red/green/blue shift
        )
        server_init = (
            struct.pack(">HH", self._fb_width, self._fb_height)
            + pixel_format
            + struct.pack(">I", len(_SERVER_NAME))
            + _SERVER_NAME
        )
        w.write(server_init)
        await w.drain()
        logger.debug("handshake: sent ServerInit (%dx%d, name=%r)", self._fb_width, self._fb_height, _SERVER_NAME)

    async def _client_recv_loop(self, client: _VncClient) -> None:
        r = client.reader
        while True:
            msg_type_b = await r.readexactly(1)
            msg_type = msg_type_b[0]
            if msg_type == 0:
                # SetPixelFormat: padding(3) + pixel-format(16). We
                # always send pixels in the format ServerInit advertised
                # (32-bit BGRX); we don't honor the client's preferred
                # format. Log it so we can see whether mac Screen
                # Sharing tried to switch us to something else.
                await r.readexactly(3)
                pf = await r.readexactly(16)
                bpp, depth, big, true_c, rmax, gmax, bmax, rshift, gshift, bshift = struct.unpack(
                    ">BBBB HHH BBB", pf[:13]
                )
                logger.debug(
                    "SetPixelFormat ignored: bpp=%d depth=%d big=%d true_c=%d max=(%d,%d,%d) shift=(%d,%d,%d)",
                    bpp,
                    depth,
                    big,
                    true_c,
                    rmax,
                    gmax,
                    bmax,
                    rshift,
                    gshift,
                    bshift,
                )
            elif msg_type == 2:
                # SetEncodings: padding(1) + n(2) + n * int32
                await r.readexactly(1)
                n = struct.unpack(">H", await r.readexactly(2))[0]
                raw = await r.readexactly(4 * n)
                client.encodings = list(struct.unpack(f">{n}i", raw))
                client.extended_clipboard = (
                    self._clipboard_monitor is not None and rfb_clipboard.ENC_EXTENDED_CLIPBOARD in client.encodings
                )
                if client.extended_clipboard:
                    # The spec wants a caps message for every SetEncodings that lists the encoding.
                    self._queue_message(client, rfb_clipboard.extended_caps())
                logger.debug(
                    "VNC client encodings: %s",
                    [
                        {
                            _ENC_RAW: "Raw",
                            _ENC_COPY_RECT: "CopyRect",
                            _ENC_CURSOR: "Cursor",
                            _ENC_DESKTOP_SIZE: "DesktopSize",
                            _ENC_LAST_RECT: "LastRect",
                        }.get(e, str(e))
                        for e in client.encodings
                    ],
                )
            elif msg_type == 3:
                # FramebufferUpdateRequest: incremental(1) + x(2) + y(2) + w(2) + h(2)
                payload = await r.readexactly(9)
                inc, fx, fy, fw, fh = struct.unpack(">BHHHH", payload)
                logger.debug(
                    "FramebufferUpdateRequest incremental=%d (%d,%d) %dx%d",
                    inc,
                    fx,
                    fy,
                    fw,
                    fh,
                )
                client.wants_update.set()
            elif msg_type == 4:
                # KeyEvent: down-flag(1) + padding(2) + key(4)  (X11 keysym).
                payload = await r.readexactly(7)
                down_flag, keysym = struct.unpack(">B2xI", payload)
                await self._handle_key(client, bool(down_flag), keysym)
            elif msg_type == 5:
                # PointerEvent: button-mask(1) + x(2) + y(2)
                data = await r.readexactly(5)
                button_mask, px, py = struct.unpack(">BHH", data)
                await self._handle_pointer(client, button_mask, px, py)
            elif msg_type == 6:
                # ClientCutText: padding(3) + length(4) + text. A negative
                # length marks an Extended Clipboard message, which a client
                # only sends after we advertised the pseudo-encoding's caps.
                await r.readexactly(3)
                ln = struct.unpack(">i", await r.readexactly(4))[0]
                if ln < 0 and not client.extended_clipboard:
                    logger.warning("VNC client sent an unsolicited extended clipboard message, terminating")
                    return
                body = await r.readexactly(abs(ln))
                if self._clipboard_monitor is not None:
                    await self._handle_cut_text(client, body, extended=ln < 0)
            else:
                logger.warning("VNC client sent unknown msg-type %d, terminating", msg_type)
                return

    # ----- shared clipboard ---------------------------------------------------
    @staticmethod
    def _queue_message(client: _VncClient, message: bytes) -> None:
        client.outbox.append(message)
        client.wants_update.set()  # wakes the send loop, which flushes the outbox first

    def _on_device_clipboard(self, content: PasteboardContent) -> None:
        # RFB only carries text; an image-only copy has nothing to offer.
        text = content.text
        if text is None:
            return
        self._clipboard_text = text
        for client in self._clients:
            if client.extended_clipboard:
                # Announce; the client asks for the text when it wants it.
                self._queue_message(client, rfb_clipboard.extended_notify())
            else:
                self._queue_message(client, rfb_clipboard.classic_server_cut_text(text))

    async def _handle_cut_text(self, client: _VncClient, body: bytes, *, extended: bool) -> None:
        assert self._clipboard_monitor is not None
        text: Optional[str] = None
        if not extended:
            text = rfb_clipboard.decode_classic_text(body)
        else:
            try:
                message = rfb_clipboard.parse_extended(body)
            except (ValueError, zlib.error) as e:
                logger.warning("dropping malformed extended clipboard message: %s", e)
                return
            if message.action == rfb_clipboard.ACTION_NOTIFY:
                if message.flags & rfb_clipboard.FORMAT_TEXT:
                    self._queue_message(client, rfb_clipboard.extended_request())
            elif message.action == rfb_clipboard.ACTION_PEEK:
                self._queue_message(client, rfb_clipboard.extended_notify(self._clipboard_text is not None))
            elif message.action == rfb_clipboard.ACTION_REQUEST:
                if message.flags & rfb_clipboard.FORMAT_TEXT and self._clipboard_text is not None:
                    self._queue_message(client, rfb_clipboard.extended_provide(self._clipboard_text))
            elif message.action == rfb_clipboard.ACTION_PROVIDE:
                text = message.text
        if not text:
            return
        try:
            await self._clipboard_monitor.set_text(text)
        except Exception as e:
            logger.warning("failed to copy the VNC client's clipboard to the device: %s", e)
            return
        # The device now holds the client's own text; don't offer it back as a device copy.
        self._clipboard_text = None
        logger.debug("clipboard: VNC client -> device (%d chars)", len(text))

    async def _client_send_loop(self, client: _VncClient) -> None:
        loop = asyncio.get_running_loop()
        expected = self._fb_width * self._fb_height * 4
        # Each framebuffer is width*height*4 (~14 MB at 1264x2752). Writing that
        # through the asyncio StreamWriter runs the copy+flush ON the event loop,
        # and at motion frame rates it starves ``sock_recv`` of the RTP socket
        # -> the kernel drops incoming media packets -> broken references
        # (mosaic) + collapsed fps the moment a client attaches during motion.
        # Send instead on a BLOCKING-style loop over a dup of the client socket
        # from a thread-pool executor: ``send``/``select`` release the GIL, so
        # the event loop keeps draining RTP while the frame goes out. dup()
        # shares the file description (and O_NONBLOCK) with asyncio's socket, so
        # we must NOT setblocking() -- we select() on the non-blocking fd. The
        # asyncio side only reads (the recv loop); nothing else writes here, so
        # a second send-direction fd is safe (TCP is full-duplex).
        base_sock = client.writer.get_extra_info("socket")
        if base_sock is None:
            return
        send_sock = socket.socket(fileno=os.dup(base_sock.fileno()))

        def _blocking_send(*chunks: bytes) -> None:
            for chunk in chunks:
                mv = memoryview(chunk)
                total = 0
                n = len(chunk)
                while total < n:
                    try:
                        total += send_sock.send(mv[total:])
                    except BlockingIOError:
                        select.select([], [send_sock], [], 2.0)

        rect_header = struct.pack(">HHHHi", 0, 0, self._fb_width, self._fb_height, _ENC_RAW)
        header = b"\x00\x00" + struct.pack(">H", 1) + rect_header
        try:
            while True:
                await client.wants_update.wait()
                client.wants_update.clear()
                if client.outbox:
                    messages, client.outbox = client.outbox, []
                    try:
                        await loop.run_in_executor(None, _blocking_send, *messages)
                    except (ConnectionResetError, BrokenPipeError, OSError):
                        return
                frame = self._latest_frame
                if frame is None or frame is client.last_sent_frame:
                    continue
                if len(frame) != expected:
                    # SPS-locked dimensions shouldn't change mid-session, but if
                    # VT ever hands us a differently sized buffer we skip rather
                    # than desync the TCP stream.
                    logger.warning(
                        "frame size mismatch: got %d bytes, fb expects %d (%dx%d * 4) -- dropping",
                        len(frame),
                        expected,
                        self._fb_width,
                        self._fb_height,
                    )
                    continue
                client.last_sent_frame = frame
                try:
                    await loop.run_in_executor(None, _blocking_send, header, frame)
                except (ConnectionResetError, BrokenPipeError, OSError):
                    return
        finally:
            with contextlib.suppress(OSError):
                send_sock.close()

    # ----- top-level orchestration ------------------------------------------
    async def serve(self) -> None:
        # 1) - 4) Device stream, audio and the background tasks (see ScreenMirrorSession.start).
        await self.start()
        loop = asyncio.get_running_loop()

        # 5) TCP listener for VNC clients. If it cannot listen, release the device stream.
        try:
            server = await asyncio.start_server(self._handle_client, self._bind, self._port)
        except BaseException:
            await self.stop()
            raise
        stop_event = asyncio.Event()

        def _request_stop() -> None:
            if not stop_event.is_set():
                logger.info("shutting down...")
                stop_event.set()

        import signal as _signal

        for signame in ("SIGINT", "SIGTERM"):
            with contextlib.suppress(NotImplementedError, AttributeError):
                loop.add_signal_handler(getattr(_signal, signame), _request_stop)

        if self._clipboard_monitor is not None:
            await self._clipboard_monitor.start()
        serve_task = asyncio.create_task(server.serve_forever())
        try:
            logger.info(
                "VNC ready: connect with `vnc://%s:%d` (Finder Cmd+K). Ctrl-C to stop.",
                self._bind,
                self._port,
            )
            await stop_event.wait()
        except (KeyboardInterrupt, asyncio.CancelledError):
            pass
        finally:
            if not serve_task.done():
                serve_task.cancel()
            logger.debug("shutdown: closing TCP listener")
            server.close()
            with contextlib.suppress(Exception):
                await asyncio.wait_for(server.wait_closed(), timeout=2.0)
            if self._clipboard_monitor is not None:
                logger.debug("shutdown: stopping clipboard monitor")
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(self._clipboard_monitor.stop(), timeout=3.0)
            await self.stop()
            current = asyncio.current_task()
            stragglers = [t for t in asyncio.all_tasks(loop) if t is not current and not t.done()]
            for t in stragglers:
                t.cancel()
            if stragglers:
                with contextlib.suppress(Exception):
                    await asyncio.wait(stragglers, timeout=2.0)
            logger.info("shutdown complete")
