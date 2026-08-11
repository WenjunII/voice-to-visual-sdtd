import threading
import unittest

from pythonosc import udp_client

from osc_control import OscControlServer, normalize_control_value


class OscControlTests(unittest.TestCase):
    def test_normalizes_keyboard_aliases_and_auto_language(self):
        self.assertEqual(normalize_control_value("visual_mode", "x"), "asian_black_brown")
        self.assertEqual(normalize_control_value("prompt_style", "general scene"), "general_scene")
        self.assertIsNone(normalize_control_value("language", "auto"))

    def test_receives_a_loopback_control_message(self):
        received = []
        ready = threading.Event()

        def on_control(name, value):
            received.append((name, value))
            ready.set()

        server = OscControlServer("127.0.0.1", 0, on_control)
        address = server.start()
        self.assertEqual(server.thread.name, "voice-to-visual-osc-control")
        self.assertFalse(server.thread.daemon)
        client = None
        try:
            client = udp_client.SimpleUDPClient(*address)
            client.send_message("/control/language", "chinese")
            self.assertTrue(ready.wait(timeout=2.0))
        finally:
            if client is not None:
                client._sock.close()
            server.stop()

        self.assertEqual(received, [("language", "zh")])


if __name__ == "__main__":
    unittest.main()
