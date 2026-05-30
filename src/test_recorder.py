import os
import tempfile
from recorder import EpisodeRecorder

def test_episode_recorder_save_load():
    recorder = EpisodeRecorder()
    recorder.record_transition(observation="obs1", action="a1", reward=1.0, info={"step": 1})
    recorder.record_transition(observation="obs2", action="a2", reward=2.0)
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp_path = tmp.name
    try:
        recorder.save_to_jsonl(tmp_path)
        # Reset and load
        recorder.clear()
        assert len(recorder.transitions) == 0
        recorder.load_from_jsonl(tmp_path)
        assert len(recorder.transitions) == 2
        assert recorder.transitions[0]["observation"] == "obs1"
        assert recorder.transitions[1]["reward"] == 2.0
        assert "info" in recorder.transitions[0]
    finally:
        os.remove(tmp_path)

if __name__ == "__main__":
    test_episode_recorder_save_load()
    print("EpisodeRecorder save/load test passed.")
