from __future__ import annotations

import unittest

from ttsd.models.microsteps import build_microstep_schedule


class MicrostepScheduleTest(unittest.TestCase):
    def test_without_extras_matches_base_length(self) -> None:
        schedule = build_microstep_schedule(base_num_steps=50, extra_steps_by_step={}, index_base=1)

        self.assertEqual(schedule.effective_num_steps, 50)
        self.assertAlmostEqual(schedule.sigmas[0], 1.0)
        self.assertAlmostEqual(schedule.sigmas[-1], 0.02098)
        self.assertEqual(schedule.base_to_refined_start[1], 0)
        self.assertEqual(schedule.base_to_refined_start[50], 49)

    def test_inserts_after_one_based_step(self) -> None:
        schedule = build_microstep_schedule(base_num_steps=50, extra_steps_by_step={40: 5}, index_base=1)

        self.assertEqual(schedule.effective_num_steps, 55)
        self.assertEqual(schedule.base_to_refined_start[40], 39)
        self.assertEqual(schedule.base_to_refined_start[41], 45)

        high = schedule.sigmas[39]
        low = schedule.sigmas[45]
        inserted = schedule.sigmas[40:45]
        self.assertEqual(len(inserted), 5)
        self.assertTrue(all(high > sigma > low for sigma in inserted))
        self.assertEqual(inserted, sorted(inserted, reverse=True))

    def test_combination_shifts_later_base_indices(self) -> None:
        schedule = build_microstep_schedule(
            base_num_steps=50,
            extra_steps_by_step={10: 10, 40: 5},
            index_base=1,
        )

        self.assertEqual(schedule.effective_num_steps, 65)
        self.assertEqual(schedule.base_to_refined_start[10], 9)
        self.assertEqual(schedule.base_to_refined_start[11], 20)
        self.assertEqual(schedule.base_to_refined_start[40], 49)
        self.assertEqual(schedule.base_to_refined_start[41], 55)

    def test_rejects_out_of_range_step(self) -> None:
        with self.assertRaisesRegex(ValueError, "outside"):
            build_microstep_schedule(base_num_steps=50, extra_steps_by_step={51: 5}, index_base=1)


if __name__ == "__main__":
    unittest.main()
