from collections import defaultdict

frame_counts_level_1_or_higher = defaultdict(int)
frame_counts_level_2_or_higher = defaultdict(int)
frame_counts_level_3 = defaultdict(int)

# high_risk_frame_counts = defaultdict(int)
CONFIRMATION_FRAMES = 10

def determine_alert_level(track_id, crossing_probability, risk_score):
    """Function to determine the alert level based on crossing probability"""
    if crossing_probability < 0.40 or risk_score < 0.30:
        alert_level = 0
    elif crossing_probability >= 0.40 and risk_score < 0.50:
        alert_level = 1
        frame_counts_level_1_or_higher[track_id] += 1
        frame_counts_level_2_or_higher[track_id] = 0
        frame_counts_level_3[track_id] = 0
    elif crossing_probability >= 0.65 and risk_score >= 0.50:
        alert_level = 2
        frame_counts_level_1_or_higher[track_id] += 1
        frame_counts_level_2_or_higher[track_id] += 1
        frame_counts_level_3[track_id] = 0
    elif crossing_probability >= 0.80 and risk_score >= 0.70:
        alert_level = 3
        frame_counts_level_1_or_higher[track_id] += 1
        frame_counts_level_2_or_higher[track_id] += 1
        frame_counts_level_3[track_id] += 1
    else:
        alert_level = 0

    if alert_level == 0:
        frame_counts_level_1_or_higher[track_id] = 0
        frame_counts_level_2_or_higher[track_id] = 0
        frame_counts_level_3[track_id] = 0

    if frame_counts_level_3[track_id] >= CONFIRMATION_FRAMES:
        confirmed_alert_level = 3
    elif frame_counts_level_2_or_higher[track_id] >= CONFIRMATION_FRAMES:
        confirmed_alert_level = 2
    elif frame_counts_level_1_or_higher[track_id] >= CONFIRMATION_FRAMES:
        confirmed_alert_level = 1
    else:
        confirmed_alert_level = 0

    # if risk_score > 0.55:
    #     high_risk_frame_counts[track_id] += 1
    # else:
    #     high_risk_frame_counts[track_id] = 0

    # print(f"DEBUG: track_id {track_id} high_risk_frame_count = {high_risk_frame_counts[track_id]}")
    # confirmed = high_risk_frame_counts[track_id] >= CONFIRMATION_FRAMES
    
    return alert_level, confirmed_alert_level

ALERT_MESSAGES = {
    0: ("No warning", None),
    1: ("Steady yellow light", None),
    2: ("Flashing orange light", "short_tone"),
    3: ("Flashing red light", "urgent_tone")
}

def get_alert_output(alert_level, confirmed):
    if not confirmed:
        return ALERT_MESSAGES[0]
    return ALERT_MESSAGES[alert_level]