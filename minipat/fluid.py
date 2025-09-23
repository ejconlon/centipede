from __future__ import annotations

import argparse
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path

SOUNDFONT_URL = "https://launchpad.net/ubuntu/+archive/primary/+sourcefiles/fluid-soundfont/3.1-5.3/fluid-soundfont_3.1.orig.tar.gz"
DEFAULT_SF_NAME = "default.sf2"
DEFAULT_PORT = "minipat"


class FluidSynthConfig:
    def __init__(self) -> None:
        self.data_dir = Path.home() / ".local" / "share" / "minipat"
        self.config_dir = Path.home() / ".local" / "config" / "minipat"
        self.sf_dir = self.data_dir / "sf"
        self.config_file = self.config_dir / "fluidsynth.cfg"
        self.nvim_config_file = self.config_dir / "nvim_config.lua"

    def setup(self, port: str, overwrite: bool) -> None:
        self.setup_soundfonts()
        self.create_config(port=port, overwrite=overwrite)

    def setup_soundfonts(self) -> None:
        self._create_directories()
        self._download_soundfont()
        self._create_symlink()

    def create_config(self, port: str, overwrite: bool) -> None:
        self.config_dir.mkdir(parents=True, exist_ok=True)
        if not overwrite and self.config_file.exists():
            print(f"Config file already exists: {self.config_file}")
            print("Use --overwrite to replace it")
            return
        self._create_config(port)
        self._create_nvim_config(port)

    def _create_directories(self) -> None:
        self.sf_dir.mkdir(parents=True, exist_ok=True)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created soundfont directory: {self.sf_dir}")
        print(f"Created config directory: {self.config_dir}")

    def _download_soundfont(self) -> None:
        # Use temp directory for tar.gz file to enable caching across runs
        temp_dir = Path(tempfile.gettempdir())
        tar_path = temp_dir / "fluid-soundfont.tar.gz"
        extract_dir = self.sf_dir / "fluid-soundfont-3.1"
        soundfont_file = extract_dir / "FluidR3_GM.sf2"

        if soundfont_file.exists():
            print(f"Soundfont already exists: {soundfont_file}")
            return

        if not tar_path.exists():
            print(f"Downloading fluid soundfont from {SOUNDFONT_URL}...")
            urllib.request.urlretrieve(SOUNDFONT_URL, tar_path)
            print(f"Downloaded to {tar_path}")
        else:
            print(f"Using existing archive: {tar_path}")

        if not extract_dir.exists():
            print("Extracting soundfont archive...")
            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extractall(self.sf_dir, filter="data")
            print(f"Extracted to {extract_dir}")

        # Keep tar.gz file in temp directory for future use (don't delete)

    def _create_symlink(self) -> None:
        source_sf = self.sf_dir / "fluid-soundfont-3.1" / "FluidR3_GM.sf2"
        target_sf = self.sf_dir / DEFAULT_SF_NAME

        if target_sf.exists() or target_sf.is_symlink():
            print(f"Default symlink already exists: {target_sf}")
            return

        if source_sf.exists():
            target_sf.symlink_to(source_sf)
            print(f"Created symlink: {target_sf} -> {source_sf}")
        else:
            print(f"Warning: Source soundfont not found at {source_sf}")

    def _create_config(self, port: str) -> None:
        # Backup existing config if it exists
        if self.config_file.exists():
            backup_file = Path(str(self.config_file) + ".bak")
            shutil.copy2(self.config_file, backup_file)
            print(f"Backed up existing config to: {backup_file}")

        config_content = f"""set midi.portname {port}
set midi.autoconnect 1
set synth.default-soundfont {self.sf_dir / DEFAULT_SF_NAME}
"""

        with open(self.config_file, "w") as f:
            f.write(config_content)
        print(f"Created FluidSynth config: {self.config_file}")

    def _create_nvim_config(self, port: str) -> None:
        # Backup existing nvim config if it exists
        if self.nvim_config_file.exists():
            backup_file = Path(str(self.nvim_config_file) + ".bak")
            shutil.copy2(self.nvim_config_file, backup_file)
            print(f"Backed up existing nvim config to: {backup_file}")

        nvim_config_content = f"""return {{
  config = {{
    minipat = {{
      port = "{port}",
    }},
    backend = {{
      command = "fluidsynth -qsi -f {self.config_file}",
    }},
  }},
}}
"""

        with open(self.nvim_config_file, "w") as f:
            f.write(nvim_config_content)
        print(f"Created nvim config: {self.nvim_config_file}")

    def get_soundfont_path(self) -> Path:
        return self.sf_dir / DEFAULT_SF_NAME

    def list_soundfonts(self) -> list[Path]:
        if not self.sf_dir.exists():
            return []
        return list(self.sf_dir.glob("*.sf2"))


def get_default_soundfont() -> Path:
    config = FluidSynthConfig()
    return config.get_soundfont_path()


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m minipat.fluid",
        description="Manage FluidSynth configuration and soundfonts for minipat",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Config command
    config_parser = subparsers.add_parser(
        "config",
        help="Create FluidSynth configuration file",
    )
    config_parser.add_argument(
        "-p",
        "--port",
        default=DEFAULT_PORT,
        help=f"MIDI port name (default: {DEFAULT_PORT})",
    )
    config_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing configuration file",
    )

    # Soundfonts command
    subparsers.add_parser(
        "soundfonts",
        help="Download and setup soundfonts",
    )

    # Setup command (does both)
    setup_parser = subparsers.add_parser(
        "setup",
        help="Setup both config and soundfonts (equivalent to running both commands)",
    )
    setup_parser.add_argument(
        "-p",
        "--port",
        default=DEFAULT_PORT,
        help=f"MIDI port name (default: {DEFAULT_PORT})",
    )
    setup_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing configuration file",
    )

    # List command
    subparsers.add_parser(
        "list",
        help="List available soundfonts",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Get port if available
    port = getattr(args, "port", DEFAULT_PORT)
    overwrite = getattr(args, "overwrite", False)
    config = FluidSynthConfig()

    if args.command == "config":
        print(f"Creating FluidSynth configuration with port: {port}")
        config.create_config(port=port, overwrite=overwrite)

    elif args.command == "soundfonts":
        print("Setting up soundfonts...")
        config.setup_soundfonts()
        print("Soundfonts setup complete!")

    elif args.command == "setup":
        print(f"Running full setup with port: {port}")
        config.setup_soundfonts()
        config.create_config(port=port, overwrite=overwrite)
        print("Full setup complete!")

    elif args.command == "list":
        soundfonts = config.list_soundfonts()
        if soundfonts:
            print("Available soundfonts:")
            for sf in soundfonts:
                is_default = " (default)" if sf.name == DEFAULT_SF_NAME else ""
                print(f"  - {sf.name}{is_default}")
        else:
            print(
                "No soundfonts found. Run 'python -m minipat.fluid soundfonts' to download."
            )


if __name__ == "__main__":
    main()
