import re
import unittest
from pathlib import Path

from dependency_profiles import (
    BACKEND_REQUIREMENT_PROFILES,
    install_command_for_backend,
    requirement_profile_for_backend,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolved_requirements(relative_path, active=None):
    path = (PROJECT_ROOT / relative_path).resolve()
    active = set() if active is None else active
    if path in active:
        raise AssertionError(f"cyclic requirement include: {path}")
    if not path.is_file():
        raise AssertionError(f"missing requirement file: {path}")

    active.add(path)
    packages = set()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("-r "):
            included = (path.parent / line[3:].strip()).resolve()
            included_relative = included.relative_to(PROJECT_ROOT)
            packages.update(resolved_requirements(included_relative, active))
            continue
        name = re.split(r"[<>=!~\[]", line, maxsplit=1)[0]
        packages.add(name.strip().lower().replace("_", "-"))
    active.remove(path)
    return packages


class DependencyProfileTests(unittest.TestCase):
    def test_every_requirement_profile_resolves_without_cycles(self):
        profiles = sorted((PROJECT_ROOT / "requirements").glob("*.txt"))

        self.assertTrue(profiles)
        for profile in profiles:
            with self.subTest(profile=profile.name):
                self.assertTrue(
                    resolved_requirements(profile.relative_to(PROJECT_ROOT))
                )

    def test_every_backend_maps_to_an_existing_profile_and_command(self):
        self.assertEqual(
            set(BACKEND_REQUIREMENT_PROFILES),
            {"whisper", "faster_whisper", "groq", "groq_hybrid", "google"},
        )
        for backend, relative_path in BACKEND_REQUIREMENT_PROFILES.items():
            with self.subTest(backend=backend):
                self.assertTrue((PROJECT_ROOT / relative_path).is_file())
                self.assertEqual(
                    install_command_for_backend(backend),
                    f"python -m pip install -r {relative_path}",
                )

    def test_profiles_resolve_the_expected_backend_dependencies(self):
        expected = {
            "whisper": {"openai-whisper", "torch", "silero-vad"},
            "faster_whisper": {
                "faster-whisper",
                "ctranslate2",
                "torch",
                "silero-vad",
            },
            "groq": set(),
            "groq_hybrid": {"argostranslate", "langdetect"},
            "google": {"speechrecognition"},
        }
        core = resolved_requirements("requirements/core.txt")
        self.assertEqual(
            core,
            {"numpy", "pyaudio", "python-osc", "requests", "transformers"},
        )

        for backend, backend_packages in expected.items():
            with self.subTest(backend=backend):
                packages = resolved_requirements(
                    requirement_profile_for_backend(backend)
                )
                self.assertTrue(core.issubset(packages))
                self.assertEqual(packages - core, backend_packages)

    def test_visual_runtime_packages_are_isolated_from_bridge_profiles(self):
        visual_packages = resolved_requirements(
            "requirements/streamdiffusion.txt"
        )
        visual_only = {
            "streamdiffusion",
            "torchvision",
            "torchaudio",
            "cupy-cuda12x",
            "spoutgl",
        }
        self.assertTrue(visual_only.issubset(visual_packages))

        for relative_path in BACKEND_REQUIREMENT_PROFILES.values():
            with self.subTest(profile=relative_path):
                self.assertTrue(
                    visual_only.isdisjoint(resolved_requirements(relative_path))
                )

    def test_legacy_requirements_file_selects_the_recommended_profile(self):
        requirements = (PROJECT_ROOT / "requirements.txt").read_text(
            encoding="utf-8"
        )

        self.assertIn("-r requirements/faster-whisper.txt", requirements)
        self.assertEqual(
            resolved_requirements("requirements.txt"),
            resolved_requirements("requirements/faster-whisper.txt"),
        )

    def test_unknown_backend_has_an_actionable_error(self):
        with self.assertRaisesRegex(
            ValueError,
            "Unsupported transcription backend: unknown",
        ):
            requirement_profile_for_backend("unknown")


if __name__ == "__main__":
    unittest.main()
