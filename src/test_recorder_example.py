from recorder import EpisodeRecorder

if __name__ == "__main__":
    # Create recorder
    recorder = EpisodeRecorder()
    # Record a few transitions manually
    recorder.record_transition("obs0", "action0", 0.0, {"step": 0})
    recorder.record_transition("obs1", "action1", 1.0, {"step": 1})
    recorder.record_transition("obs2", "action2", 0.5, {"step": 2})

    print("Transitions recorded:")
    for t in recorder.transitions:
        print(t)

    summary = recorder.episode_summary()
    print("Episode summary:")
    print(summary)
    # Expected: length 3, total_reward 1.5, actions ['action0', 'action1', 'action2']
