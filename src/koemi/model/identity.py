from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import hashlib
import json
import re
import unicodedata
from typing import Any, Literal


IDENTITY_FORMAT_VERSION = 1
IdentityAvailability = Literal["open", "restricted", "closed", "unknown"]
_IDENTITY_AVAILABILITIES = frozenset({"open", "restricted", "closed", "unknown"})
_CONTROL_CHARACTER = re.compile(r"[\x00-\x1f\x7f]")


def _require_text(value: Any, field_name: str, maximum_bytes: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"identity field '{field_name}' must be a non-empty string")
    normalized = value.strip()
    if _CONTROL_CHARACTER.search(normalized):
        raise ValueError(f"identity field '{field_name}' contains a control character")
    if len(normalized.encode("utf-8")) > maximum_bytes:
        raise ValueError(f"identity field '{field_name}' exceeds {maximum_bytes} UTF-8 bytes")
    return normalized


def _optional_text(value: Any, field_name: str, maximum_bytes: int) -> str | None:
    if value is None:
        return None
    return _require_text(value, field_name, maximum_bytes)


def _validate_availability(value: Any, field_name: str) -> IdentityAvailability:
    if value not in _IDENTITY_AVAILABILITIES:
        raise ValueError(f"identity field '{field_name}' has an unsupported availability")
    return value


def _slug(value: str) -> str:
    ascii_value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", ascii_value).strip("-").lower()
    if slug:
        return slug
    return f"model-{hashlib.sha256(value.encode('utf-8')).hexdigest()[:12]}"


def _require_mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"identity field '{field_name}' must be an object")
    return value


@dataclass(frozen=True)
class ModelLicensing:
    source_license: str | None = None
    weights_license: str | None = None
    data_license: str | None = None
    source_availability: IdentityAvailability = "closed"
    weights_availability: IdentityAvailability = "closed"
    data_availability: IdentityAvailability = "unknown"
    license_url: str | None = None
    attribution: str | None = None

    def __post_init__(self) -> None:
        for field_name in ("source_availability", "weights_availability", "data_availability"):
            _validate_availability(getattr(self, field_name), field_name)
        for field_name in ("source_license", "weights_license", "data_license"):
            value = getattr(self, field_name)
            if value is not None:
                _require_text(value, field_name, 128)
        _optional_text(self.license_url, "license_url", 512)
        _optional_text(self.attribution, "attribution", 512)
        for artifact_name, availability, license_name in (
            ("source", self.source_availability, self.source_license),
            ("weights", self.weights_availability, self.weights_license),
            ("data", self.data_availability, self.data_license),
        ):
            if availability == "open" and license_name is None:
                raise ValueError(f"open {artifact_name} availability requires a license")

    def to_payload(self) -> dict[str, Any]:
        return {
            "source_license": self.source_license,
            "weights_license": self.weights_license,
            "data_license": self.data_license,
            "source_availability": self.source_availability,
            "weights_availability": self.weights_availability,
            "data_availability": self.data_availability,
            "license_url": self.license_url,
            "attribution": self.attribution,
        }

    @classmethod
    def from_payload(cls, payload: Any) -> ModelLicensing:
        values = _require_mapping(payload, "licensing")
        expected = {
            "source_license",
            "weights_license",
            "data_license",
            "source_availability",
            "weights_availability",
            "data_availability",
            "license_url",
            "attribution",
        }
        if set(values) != expected:
            raise ValueError("identity licensing payload has an unexpected schema")
        return cls(**dict(values))


@dataclass(frozen=True)
class ModelIdentity:
    organization: str
    model_name: str
    version: str
    architecture: str = "HERM"
    heading: str | None = None
    variant: str | None = None
    description: str | None = None
    licensing: ModelLicensing = field(default_factory=ModelLicensing)

    def __post_init__(self) -> None:
        organization = _require_text(self.organization, "organization", 160)
        model_name = _require_text(self.model_name, "model_name", 160)
        version = _require_text(self.version, "version", 80)
        architecture = _require_text(self.architecture, "architecture", 80)
        variant = _optional_text(self.variant, "variant", 120)
        description = _optional_text(self.description, "description", 2048)
        heading = self.heading
        if heading is None:
            heading = f"{organization} {model_name} ({version})"
            if variant is not None:
                heading = f"{heading} [{variant}]"
        heading = _require_text(heading, "heading", 240)
        if not isinstance(self.licensing, ModelLicensing):
            raise TypeError("identity licensing must be a ModelLicensing instance")
        object.__setattr__(self, "organization", organization)
        object.__setattr__(self, "model_name", model_name)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "architecture", architecture)
        object.__setattr__(self, "heading", heading)
        object.__setattr__(self, "variant", variant)
        object.__setattr__(self, "description", description)

    @property
    def model_id(self) -> str:
        suffix = f"-{_slug(self.variant)}" if self.variant is not None else ""
        return f"{_slug(self.organization)}/{_slug(self.model_name)}:{_slug(self.version)}{suffix}"

    @property
    def identity_digest(self) -> str:
        encoded = json.dumps(self.to_payload(), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def to_payload(self) -> dict[str, Any]:
        return {
            "format_version": IDENTITY_FORMAT_VERSION,
            "organization": self.organization,
            "model_name": self.model_name,
            "version": self.version,
            "architecture": self.architecture,
            "heading": self.heading,
            "variant": self.variant,
            "description": self.description,
            "licensing": self.licensing.to_payload(),
        }

    @classmethod
    def from_payload(cls, payload: Any) -> ModelIdentity:
        values = _require_mapping(payload, "identity")
        expected = {
            "format_version",
            "organization",
            "model_name",
            "version",
            "architecture",
            "heading",
            "variant",
            "description",
            "licensing",
        }
        if set(values) != expected:
            raise ValueError("identity payload has an unexpected schema")
        if values["format_version"] != IDENTITY_FORMAT_VERSION:
            raise ValueError("identity format version is not supported")
        return cls(
            organization=values["organization"],
            model_name=values["model_name"],
            version=values["version"],
            architecture=values["architecture"],
            heading=values["heading"],
            variant=values["variant"],
            description=values["description"],
            licensing=ModelLicensing.from_payload(values["licensing"]),
        )

    def render_header(self) -> str:
        lines = [
            "HERM-MODEL-IDENTITY/1",
            f"heading={self.heading}",
            f"model_id={self.model_id}",
            f"organization={self.organization}",
            f"model_name={self.model_name}",
            f"architecture={self.architecture}",
            f"version={self.version}",
            f"variant={self.variant or ''}",
            f"source={self.licensing.source_availability}:{self.licensing.source_license or ''}",
            f"weights={self.licensing.weights_availability}:{self.licensing.weights_license or ''}",
            f"data={self.licensing.data_availability}:{self.licensing.data_license or ''}",
            f"identity_digest={self.identity_digest}",
        ]
        return "\n".join(lines)


def attach_model_identity(model: Any, identity: ModelIdentity) -> Any:
    if not isinstance(identity, ModelIdentity):
        raise TypeError("model identity must be a ModelIdentity instance")
    try:
        setattr(model, "model_identity", identity)
    except (AttributeError, TypeError) as error:
        raise TypeError("model does not accept an identity binding") from error
    return model


def get_model_identity(model: Any) -> ModelIdentity | None:
    identity = getattr(model, "model_identity", None)
    if identity is not None and not isinstance(identity, ModelIdentity):
        raise TypeError("model identity binding is invalid")
    return identity


__all__ = [
    "IDENTITY_FORMAT_VERSION",
    "IdentityAvailability",
    "ModelIdentity",
    "ModelLicensing",
    "attach_model_identity",
    "get_model_identity",
]
