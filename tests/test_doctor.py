from __future__ import annotations

import unittest

from dwhisper.doctor import DoctorCheck, run_doctor, summarize, worst_status


class DoctorTests(unittest.TestCase):
    def test_run_doctor_returns_one_entry_per_check(self) -> None:
        calls: list[str] = []

        def make_check(name: str, status: str) -> DoctorCheck:
            def check() -> DoctorCheck:
                calls.append(name)
                return DoctorCheck(name=name, status=status, message=f"{name}: {status}")
            return check

        results = run_doctor(
            checks=(
                make_check("a", "ok"),
                make_check("b", "warn"),
                make_check("c", "error"),
            )
        )

        self.assertEqual(calls, ["a", "b", "c"])
        self.assertEqual([r.name for r in results], ["a", "b", "c"])
        self.assertEqual([r.status for r in results], ["ok", "warn", "error"])

    def test_failing_check_does_not_abort_doctor(self) -> None:
        def boom() -> DoctorCheck:
            raise RuntimeError("bad")

        def fine() -> DoctorCheck:
            return DoctorCheck(name="fine", status="ok", message="ok")

        results = run_doctor(checks=(boom, fine))

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].status, "error")
        self.assertIn("bad", results[0].message)
        self.assertEqual(results[1].status, "ok")

    def test_summary_and_worst_status_reflect_results(self) -> None:
        results = [
            DoctorCheck(name="a", status="ok", message=""),
            DoctorCheck(name="b", status="warn", message=""),
            DoctorCheck(name="c", status="ok", message=""),
            DoctorCheck(name="d", status="error", message=""),
        ]

        summary = summarize(results)
        self.assertEqual(summary["ok"], 2)
        self.assertEqual(summary["warn"], 1)
        self.assertEqual(summary["error"], 1)
        self.assertEqual(worst_status(results), "error")

    def test_worst_status_handles_all_ok(self) -> None:
        results = [
            DoctorCheck(name="a", status="ok", message=""),
            DoctorCheck(name="b", status="ok", message=""),
        ]
        self.assertEqual(worst_status(results), "ok")


if __name__ == "__main__":
    unittest.main()
