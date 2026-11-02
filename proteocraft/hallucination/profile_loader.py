"""Load versioned ProteoCraft launch profiles into an argparse parser."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


PROFILE_SCHEMA_VERSION = 1
PROFILE_DIRECTORY = (
    Path(__file__).resolve().parents[1] / "configs" / "proteocraft_profiles"
)
PROFILE_METADATA_KEYS = frozenset(
    {"schema_version", "name", "description", "defaults"}
)


class LaunchProfileError(ValueError):
    """Raised when a launch profile is missing or invalid."""


def available_launch_profiles() -> tuple[str, ...]:
    """Return bundled profile names in deterministic order."""

    if not PROFILE_DIRECTORY.is_dir():
        return ()
    return tuple(path.stem for path in sorted(PROFILE_DIRECTORY.glob("*.yaml")))


def resolve_launch_profile_path(profile: str | Path) -> Path:
    """Resolve a bundled profile name or an explicit YAML path."""

    raw_profile = str(profile).strip()
    if not raw_profile:
        raise LaunchProfileError("launch profile must not be empty")

    candidate = Path(raw_profile).expanduser()
    if candidate.suffix.lower() not in {".yaml", ".yml"} and len(candidate.parts) == 1:
        candidate = PROFILE_DIRECTORY / f"{raw_profile}.yaml"
    elif not candidate.is_absolute():
        candidate = Path.cwd() / candidate

    candidate = candidate.resolve()
    if not candidate.is_file():
        bundled = ", ".join(available_launch_profiles()) or "<none>"
        raise LaunchProfileError(
            f"launch profile not found: {candidate}; bundled profiles: {bundled}"
        )
    return candidate


def load_launch_profile(profile: str | Path) -> tuple[Path, dict[str, Any]]:
    """Read and validate the profile envelope before argparse validation."""

    profile_path = resolve_launch_profile_path(profile)
    with profile_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)

    if not isinstance(payload, dict):
        raise LaunchProfileError(
            f"launch profile must contain a YAML mapping: {profile_path}"
        )

    unknown_metadata = sorted(set(payload) - PROFILE_METADATA_KEYS)
    if unknown_metadata:
        raise LaunchProfileError(
            "unknown launch profile metadata keys: "
            + ", ".join(unknown_metadata)
        )

    schema_version = payload.get("schema_version")
    if schema_version != PROFILE_SCHEMA_VERSION:
        raise LaunchProfileError(
            f"unsupported launch profile schema_version={schema_version!r}; "
            f"expected {PROFILE_SCHEMA_VERSION}"
        )

    profile_name = payload.get("name")
    if not isinstance(profile_name, str) or not profile_name.strip():
        raise LaunchProfileError("launch profile name must be a non-empty string")

    description = payload.get("description", "")
    if not isinstance(description, str):
        raise LaunchProfileError("launch profile description must be a string")

    defaults = payload.get("defaults")
    if not isinstance(defaults, dict) or not defaults:
        raise LaunchProfileError(
            "launch profile defaults must be a non-empty mapping"
        )
    if not all(isinstance(key, str) and key for key in defaults):
        raise LaunchProfileError("launch profile default names must be strings")

    return profile_path, payload


def _coerce_profile_value(
    action: argparse.Action,
    value: Any,
) -> Any:
    if value is None:
        return None
    if isinstance(value, (dict, list, tuple, set)):
        raise LaunchProfileError(
            f"profile value for {action.dest!r} must be a scalar"
        )

    if action.type is not None and not isinstance(value, bool):
        try:
            value = action.type(value)
        except (TypeError, ValueError, AttributeError) as exc:
            raise LaunchProfileError(
                f"invalid value for profile option {action.dest!r}: {value!r}"
            ) from exc

    if action.choices is not None and value not in action.choices:
        choices = ", ".join(str(choice) for choice in action.choices)
        raise LaunchProfileError(
            f"invalid value for profile option {action.dest!r}: {value!r}; "
            f"choose from {choices}"
        )
    return value


def apply_launch_profile(
    parser: argparse.ArgumentParser,
    profile: str | Path,
) -> tuple[Path, str, dict[str, Any]]:
    """Apply profile defaults while preserving explicit CLI precedence."""

    profile_path, payload = load_launch_profile(profile)
    actions = {
        action.dest: action
        for action in parser._actions
        if action.dest not in {"help", "profile"}
    }
    defaults = payload["defaults"]

    unknown_options = sorted(set(defaults) - set(actions))
    if unknown_options:
        raise LaunchProfileError(
            "unknown launch profile options: " + ", ".join(unknown_options)
        )

    required_options = sorted(
        option for option in defaults if actions[option].required
    )
    if required_options:
        raise LaunchProfileError(
            "runtime-required options cannot be stored in a launch profile: "
            + ", ".join(required_options)
        )

    normalized_defaults = {
        option: _coerce_profile_value(actions[option], value)
        for option, value in defaults.items()
    }
    parser.set_defaults(**normalized_defaults)
    return profile_path, payload["name"].strip(), normalized_defaults
