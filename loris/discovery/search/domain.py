"""Declared capability ablations of the real proposal axes, before BO.

The original rich factory and sampler are retained byte-for-byte. Restricted
domains are new studies: no deleted-premise rules or historical re-filtering.
"""

import hashlib
from types import MappingProxyType

from loris._engine.rlearner_config import _canonical
from .configuration import AuthorFreeCombinationSpace, build_config_space as rich_space

DOMAINS = ("rich", "no_text", "local")


def variable_names(max_variables):
    """The declared finite variable axis, not a bound imposed by the paper."""
    if type(max_variables) is not int or max_variables < 2:
        raise ValueError("max_variables must be an integer at least two")
    return tuple(("x", "y", "z")[i] if i < 3 else "v%d" % i for i in range(max_variables))


class CapabilitySpace(AuthorFreeCombinationSpace):
    def __init__(self, *, candidate_domain, **kwargs):
        if candidate_domain not in ("no_text", "local"):
            raise ValueError("restricted capability space requires no_text or local")
        if candidate_domain == "no_text":
            kwargs["patterns"] = ()
        else:
            kwargs["variables"] = ("x",)
            kwargs["relations"] = ()
        super().__init__(**kwargs)
        if candidate_domain == "local":
            features = tuple(f for f in self.features if f.kind != "label")
            object.__setattr__(self, "features", features)
            object.__setattr__(
                self, "feature_by_key", MappingProxyType({f.key: f for f in features})
            )
        manifest = self.manifest()
        manifest.pop("space_id")
        manifest["features"] = [f.to_dict() for f in self.features]
        manifest["candidate_domain"] = candidate_domain
        manifest["search"]["capability_ablation"] = {
            "no_text": "no text feature or pattern in actual proposal/initializer axes; labels and relations retained",
            "local": "one document variable; no Label body, LabelCompare, cross-document relation or relation head",
        }[candidate_domain]
        manifest["search"]["new_BO_required"] = True
        definition = _canonical(manifest)
        object.__setattr__(self, "_definition", definition)
        object.__setattr__(self, "space_id", hashlib.sha256(definition).hexdigest())


def build_config_space(*, candidate_domain="rich", **kwargs):
    if candidate_domain not in DOMAINS:
        raise ValueError("unsupported candidate_domain")
    return (
        rich_space(**kwargs)
        if candidate_domain == "rich"
        else CapabilitySpace(candidate_domain=candidate_domain, **kwargs)
    )
