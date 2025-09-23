from __future__ import annotations

from pathlib import Path

from minipat.fluid import (
    DEFAULT_SF_NAME,
    FluidSynthConfig,
    get_default_soundfont,
)


def test_fluidsynth_config_init() -> None:
    config = FluidSynthConfig()
    assert config.data_dir == Path.home() / ".local" / "share" / "minipat"
    assert config.config_dir == Path.home() / ".local" / "config" / "minipat"
    assert config.sf_dir == config.data_dir / "sf"
    assert config.config_file == config.config_dir / "fluidsynth.cfg"
    assert config.nvim_config_file == config.config_dir / "nvim_config.lua"


def test_get_soundfont_path() -> None:
    config = FluidSynthConfig()
    expected_path = config.sf_dir / DEFAULT_SF_NAME
    assert config.get_soundfont_path() == expected_path


def test_list_soundfonts() -> None:
    config = FluidSynthConfig()
    soundfonts = config.list_soundfonts()
    assert isinstance(soundfonts, list)
    if config.sf_dir.exists():
        assert all(sf.suffix == ".sf2" for sf in soundfonts)


def test_get_default_soundfont() -> None:
    sf_path = get_default_soundfont()
    assert isinstance(sf_path, Path)
    assert sf_path.name == "default.sf2"


def test_setup_creates_required_files() -> None:
    config = FluidSynthConfig()
    assert config.sf_dir.exists()
    assert config.config_file.exists()
    assert config.nvim_config_file.exists()
    assert (config.sf_dir / DEFAULT_SF_NAME).exists() or (
        config.sf_dir / DEFAULT_SF_NAME
    ).is_symlink()
