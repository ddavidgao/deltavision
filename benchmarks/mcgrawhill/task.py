"""
McGraw-Hill Connect benchmark tasks.
"""

MCGRAWHILL_TASKS = [
    {
        "id": "eaps111_ch1_quiz",
        "description": (
            "Complete Chapter 1 quiz on McGraw-Hill Connect. "
            "Answer all questions. Submit when done."
        ),
        "start_url": "https://connect.mheducation.com",
        "success_criteria": "Score page visible with percentage displayed",
        "notes": [
            "Canvas-rendered questions — no DOM text for answers",
            "URL does not change between questions",
            "Each new question = full canvas repaint = high diff_ratio",
            "Answer feedback appears as small colored overlay in corner",
            "Submit button only appears after all questions answered",
        ],
        "expected_challenges": [
            "iframe nesting around question canvas",
            "randomized element IDs per session",
            "loading spinners between questions (~800ms)",
            "drag-and-drop question types",
        ],
    }
]
