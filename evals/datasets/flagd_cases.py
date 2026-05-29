"""Suite 1 golden dataset — one Case per flagd injectable failure.

Ground truth from EXECUTION.md + OTel demo docs.
"""

from __future__ import annotations

from pydantic_evals import Case, Dataset

from agent.models import HealResult
from evals.evaluators import (
    BandFloor,
    ConfidenceLevelScore,
    EfficiencyEvaluator,
    FileMatch,
    make_diagnosis_judge,
)

# Shared evaluators applied to every case.
# LLMJudge replaces ServiceMatch + ErrorTypeMatch — judges semantically against
# the expected service + flag behavior description, avoiding the synonym problem.
def _shared_evaluators():
    return (
        make_diagnosis_judge(),
        FileMatch(),
        BandFloor(),
        ConfidenceLevelScore(),
        EfficiencyEvaluator(),
    )


def build_dataset() -> Dataset:
    """Return the flagd ground-truth dataset for end-to-end investigation evals."""
    cases = [
        Case(
            name="adFailure",
            inputs={
                "description": (
                    "ad service is throwing intermittent runtime errors. "
                    "Traces show UNAVAILABLE status returned from the ad service."
                ),
                "service_name": "ad",
            },
            expected_output={
                "service": "ad",
                "flag": "adFailure — generates a gRPC error for GetAds 1/10th of the time",
                "file_substring": "AdService",
            },
            metadata={
                "flag": "adFailure",
                "expected_band_floor": "medium",
            },
            evaluators=_shared_evaluators(),
        ),
        Case(
            name="cartFailure",
            inputs={
                "description": (
                    "cart service is throwing FailedPrecondition errors — "
                    "'Can't access cart storage'. Users can't add items to cart."
                ),
                "service_name": "cart",
            },
            expected_output={
                "service": "cart",
                "flag": "cartFailure — generates an error whenever EmptyCart is called, simulating Valkey/Redis unavailability",
                "file_substring": "ValkeyCartStore",
            },
            metadata={
                "flag": "cartFailure",
                "expected_band_floor": "medium",
            },
            evaluators=_shared_evaluators(),
        ),
        Case(
            name="paymentFailure",
            inputs={
                "description": (
                    "payment service is returning errors on checkout. "
                    "Customers are unable to complete purchases."
                ),
                "service_name": "payment",
            },
            expected_output={
                "service": "payment",
                "flag": "paymentFailure — generates an error when calling the charge method in the payment service",
                "file_substring": "payment",
            },
            metadata={
                "flag": "paymentFailure",
                "expected_band_floor": "medium",
            },
            evaluators=_shared_evaluators(),
        ),
        Case(
            name="intlShippingSlowdown",
            inputs={
                "description": (
                    "international shipping estimates are extremely slow. "
                    "Checkout latency has spiked for non-US users."
                ),
                "service_name": "shipping",
            },
            expected_output={
                "service": "shipping",
                "flag": "intlShippingSlowdown — injects artificial latency into the shipping service for international requests",
                "file_substring": "",
            },
            metadata={
                "flag": "intlShippingSlowdown",
                "expected_band_floor": "low",
            },
            evaluators=_shared_evaluators(),
        ),
        Case(
            name="adHighCpu",
            inputs={
                "description": (
                    "ad service CPU usage is extremely high. "
                    "Pods are near resource limits."
                ),
                "service_name": "ad",
            },
            expected_output={
                "service": "ad",
                "flag": "adHighCpu — triggers high CPU load in the ad service via busy-loop code",
                "file_substring": "",
            },
            metadata={
                "flag": "adHighCpu",
                "expected_band_floor": "low",
            },
            evaluators=_shared_evaluators(),
        ),
    ]

    return Dataset(name="flagd_investigation", cases=cases)
