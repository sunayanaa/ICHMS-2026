import numpy as np
import matplotlib.pyplot as plt
import random
import time
from collections import defaultdict

################################################################################
# 1. SIMULATION PARAMETERS
################################################################################

# --- World ---
WORLD_SIZE_X = 500  # World width
WORLD_SIZE_Y = 500  # World height
TARGET_POSITIONS = {
    'target_1': np.array([50, 450]),
    'target_2': np.array([250, 250]),
    'target_3': np.array([450, 50]),
}
# A "fake" target to trigger the false positive event
FAKE_TARGET_POS = np.array([450, 450]) 

# --- Swarm ---
N_AGENTS = 20
MAX_SPEED = 1.0  # units per step
AGENT_SENSOR_RANGE = 40.0 # How close to "find" a target
AGENT_SAFETY_BUFFER = 10.0 # ** INCREASED ** How close agents can get
REPULSION_STRENGTH = 1.5   # ** NEW ** How strongly agents push each other away

# --- Simulation ---
SIMULATION_STEPS = 1000  # Total time steps for one simulation run
N_SIMULATION_RUNS = 50   # How many times to run for aggregated stats (for V.A)

# --- Event Triggers (in steps) ---
EVENT_GPS_FAIL_START = 300   # When 5 agents lose GPS
EVENT_GPS_FAIL_NUM_AGENTS = 5
EVENT_FALSE_POSITIVE_START = 600 # When swarm misidentifies a target
EVENT_FALSE_POSITIVE_END = 650   # When the swarm "realizes" its mistake

# --- Human Model ---
HUMAN_REACTION_TIME_STEPS = 30 # e.g., 3 seconds if 1 step = 0.1s


################################################################################
# 2. THE AGENT CLASS
################################################################################

class Agent:
    """ Represents a single agent in the swarm. """
    def __init__(self, id):
        self.id = id
        # Start at a random position
        self.position = np.array([random.uniform(0, 100), random.uniform(0, 100)], dtype=float)
        self.velocity = np.zeros(2, dtype=float)
        
        # --- State ---
        self.gps_healthy = True
        self.local_confidence = 1.0
        self.state = "SEARCHING" # 'SEARCHING', 'FAIL_SAFE'
        self.assigned_target_key = random.choice(list(TARGET_POSITIONS.keys()))

################################################################################
# 3. CORE LOGIC FUNCTIONS (THE "BRAINS" OF YOUR PAPER)
################################################################################

def update_local_confidence(agent, current_step):
    """
    (SECTION III.B) - AGENT-SIDE
    Updates an agent's self-assessed confidence based on events.
    """
    if not agent.gps_healthy:
        # If GPS is down, our confidence in our state is low.
        agent.local_confidence = 0.1
    else:
        # GPS is fine, confidence is high.
        agent.local_confidence = 1.0

def decentralized_consensus(agents):
    """
    (SECTION III.B) - SWARM-SIDE
    Simplified consensus: The swarm's collective confidence (C_S) is
    the average of all individual agent confidences.
    """
    if not agents:
        return 0.0
    return np.mean([agent.local_confidence for agent in agents])

def run_synthetic_operator(swarm_state, current_step):
    """
    (SECTION IV.A) - HUMAN-SIDE
    This is your "synthetic operator" script. It sets the human trust (T_H)
    based on observable events and reaction times.
    """
    human_trust = 1.0
    
    # 1. React to False Positive
    if swarm_state['false_positive_detected']:
        # Human sees the mistake and immediately distrusts the swarm's
        # current action.
        return 0.0 # Force a stop
    
    # 2. React to low swarm confidence
    # ** CHANGED ** Now reacts to C_S < 0.9 (more sensitive)
    if swarm_state['swarm_confidence'] < 0.9: 
        # If this is the *first time* we see the low confidence, record the time.
        if swarm_state['low_conf_detected_at'] is None:
            swarm_state['low_conf_detected_at'] = current_step
        
        # If enough time has passed since we first saw the problem...
        if (current_step - swarm_state['low_conf_detected_at']) > HUMAN_REACTION_TIME_STEPS:
            # ...the human operator's trust drops.
            human_trust = 0.1
    else:
        # Swarm confidence is high, so reset the human's perception.
        swarm_state['low_conf_detected_at'] = None

    return human_trust

def calculate_negotiated_trust(T_H, C_S):
    """
    (SECTION III.B) - THE NEGOTIATION
    Calculates the final Negotiated Trust (T_N).
    We use min() because trust is only as high as the least confident party.
    This prioritizes safety.
    """
    return min(T_H, C_S)

#
# >>>>>>>> THIS IS THE UPDATED FUNCTION <<<<<<<<<<
#
def decentralized_control_law(agent, T_N, swarm_state, all_agents):
    """
    (SECTION III.C) - THE CONTROL LAW
    This is the core of the agent's behavior. T_N directly modulates
    the agent's speed and state.
    
    ** V2 Update: Now includes collision avoidance logic. **
    """
    
    # If trust is critically low, enter fail-safe.
    if T_N < 0.2:
        agent.state = "FAIL_SAFE"
        agent.velocity = np.zeros(2) # Stop moving
        return

    agent.state = "SEARCHING"
    
    # --- 1. Attraction Force (Move to Target) ---
    target_pos = None
    if swarm_state['false_positive_detected']:
        # The swarm is "confidently wrong"
        target_pos = FAKE_TARGET_POS
    else:
        # Move towards the correct, assigned target
        target_pos = TARGET_POSITIONS[agent.assigned_target_key]

    direction_to_target = (target_pos - agent.position)
    norm_target = np.linalg.norm(direction_to_target)
    if norm_target > 0:
        direction_to_target = direction_to_target / norm_target
    
    # --- 2. Repulsion Force (Collision Avoidance) ---
    repulsion_vector = np.zeros(2, dtype=float)
    for other in all_agents:
        if agent.id == other.id:
            continue
            
        dist = np.linalg.norm(agent.position - other.position)
        
        # If another agent is too close
        if dist < (AGENT_SAFETY_BUFFER * 2): 
            # Calculate a vector pointing *away* from the other agent
            away_vector = agent.position - other.position
            # Scale it by inverse of distance (stronger when closer)
            repulsion_vector += (away_vector / (dist**2 + 1e-6)) 

    # --- 3. Combine Forces ---
    # Give repulsion a higher weight to prioritize safety
    final_direction = (direction_to_target + 
                       (repulsion_vector * REPULSION_STRENGTH))
    
    # Normalize the final direction vector
    norm_final = np.linalg.norm(final_direction)
    if norm_final > 0:
        final_direction = final_direction / norm_final
    
    # ** THE KEY TRUST MODULATION **
    # T_N directly scales the agent's speed.
    # If T_N = 1.0, speed = MAX_SPEED.
    # If T_N = 0.5, speed = 0.5 * MAX_SPEED (slows down).
    effective_speed = MAX_SPEED * T_N
    
    agent.velocity = final_direction * effective_speed

################################################################################
# 4. THE MAIN SIMULATION RUNNER
################################################################################

def run_simulation(baseline_mode, log_time_series=False):
    """
    Runs one full simulation for a given baseline mode.
    - 'DTSC': Your proposed model
    - 'Full_Auto': Full Autonomy (T_N always 1.0, ignores safety)
    - 'Teleop': Direct Control (Ignores swarm confidence C_S)
    """
    
    # --- Initialize ---
    agents = [Agent(i) for i in range(N_AGENTS)]
    targets_found = {key: False for key in TARGET_POSITIONS}
    
    # This dictionary tracks the "ground truth" of the world
    world_state = {
        'false_positive_active': False,
    }
    
    # This dictionary tracks the *operator's perception* of the world
    operator_state = {
        'swarm_confidence': 1.0,
        'false_positive_detected': False,
        'low_conf_detected_at': None # For modeling reaction time
    }
    
    # --- Metrics to log ---
    metrics = {
        'NSV_collisions': 0,
        'CI_interventions': 0,
        'TTC': SIMULATION_STEPS # Time to Task Completion (default to max)
    }
    
    # This is for the time-series plot (V.B)
    time_series_log = []
    
    # Track collisions to avoid double counting
    collisions_this_step = set()

    # --- Main Loop ---
    for step in range(SIMULATION_STEPS):
        
        # --- 1. Event Triggers (Ground Truth) ---
        
        # GPS Failure Event
        if step == EVENT_GPS_FAIL_START:
            for i in range(EVENT_GPS_FAIL_NUM_AGENTS):
                agents[i].gps_healthy = False
        
        # False Positive Event
        world_state['false_positive_active'] = (
            step >= EVENT_FALSE_POSITIVE_START and 
            step < EVENT_FALSE_POSITIVE_END
        )
        
        # --- 2. Agent-Level Update (Confidence) ---
        for agent in agents:
            update_local_confidence(agent, step)

        # --- 3. Swarm-Level Update (Trust Negotiation) ---
        
        # Default values
        T_H = 1.0  # Human Trust
        C_S = 1.0  # Swarm Confidence
        T_N = 1.0  # Negotiated Trust

        if baseline_mode == 'DTSC':
            C_S = decentralized_consensus(agents)
            operator_state['swarm_confidence'] = C_S
            operator_state['false_positive_detected'] = world_state['false_positive_active']
            
            T_H = run_synthetic_operator(operator_state, step)
            T_N = calculate_negotiated_trust(T_H, C_S)

        elif baseline_mode == 'Full_Auto':
            # Human is not paying attention, T_H is always 1.
            # System ignores C_S. T_N is locked to 1.0.
            T_H = 1.0
            C_S = decentralized_consensus(agents) # Swarm *knows* it's failing...
            T_N = 1.0 # ...but the system doesn't care.

        elif baseline_mode == 'Teleop':
            # System ignores the swarm's self-awareness (C_S)
            C_S = 1.0 # (Effectively)
            operator_state['swarm_confidence'] = 1.0 # (Human doesn't get C_S report)
            operator_state['false_positive_detected'] = world_state['false_positive_active']

            T_H = run_synthetic_operator(operator_state, step)
            T_N = T_H # Negotiated trust = human trust
        
        # Log if the synthetic operator "intervened" (forced a low trust)
        if T_H < 0.2:
            metrics['CI_interventions'] += 1

        # --- 4. Agent-Level Update (Control & Movement) ---
        collisions_this_step.clear()
        
        for agent in agents:
            # Pass the "ground truth" to the control law for the baseline check
            control_law_state = {
                'false_positive_detected': world_state['false_positive_active']
            }
            # ** UPDATED ** Pass all_agents for collision check
            decentralized_control_law(agent, T_N, control_law_state, agents) 
            
            # Update position
            agent.position += agent.velocity
            
            # Check for collisions (NSV)
            for other in agents:
                if agent.id == other.id:
                    continue
                dist = np.linalg.norm(agent.position - other.position)
                if dist < AGENT_SAFETY_BUFFER:
                    # Use a sorted tuple to track unique collision pairs
                    pair = tuple(sorted((agent.id, other.id)))
                    collisions_this_step.add(pair)

            # Check for finding targets
            for key, pos in TARGET_POSITIONS.items():
                if np.linalg.norm(agent.position - pos) < AGENT_SENSOR_RANGE:
                    targets_found[key] = True

        metrics['NSV_collisions'] += len(collisions_this_step)
        
        # --- 5. Log Data ---
        if log_time_series:
            time_series_log.append({
                'step': step,
                'T_N': T_N,
                'T_H': T_H,
                'C_S': C_S,
                'swarm_state': agents[0].state, # Log state of first agent
                'gps_fail_active': not agents[0].gps_healthy,
                'false_positive_active': world_state['false_positive_active']
            })
        
        # Check for task completion
        if all(targets_found.values()) and metrics['TTC'] == SIMULATION_STEPS:
            metrics['TTC'] = step # Set completion time

    # We sum collisions, but only count interventions *once* per simulation
    if metrics['CI_interventions'] > 0:
        metrics['CI_interventions'] = 1
            
    return metrics, time_series_log

################################################################################
# 5. PLOTTING FUNCTIONS (FOR SECTION V)
################################################################################

def plot_bar_charts(all_results):
    """
    (FOR SECTION V.A)
    Generates the bar charts for your aggregated metrics.
    """
    print("\n--- Generating Plots for Section V.A ---")
    
    baselines = list(all_results.keys())
    
    # --- Calculate Averages ---
    avg_metrics = defaultdict(lambda: {'mean': 0, 'std': 0})
    
    for mode in baselines:
        for metric in ['TTC', 'NSV_collisions', 'CI_interventions']:
            values = [run[metric] for run in all_results[mode]]
            avg_metrics[metric][mode] = {
                'mean': np.mean(values),
                'std': np.std(values)
            }
            
    # --- Create Plots ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Simulation Results: Baseline Comparison (Avg. of {} Runs)'.format(N_SIMULATION_RUNS), y=1.02, fontsize=16)
    
    # Plot 1: Time to Task Completion (TTC)
    means = [avg_metrics['TTC'][mode]['mean'] for mode in baselines]
    stds = [avg_metrics['TTC'][mode]['std'] for mode in baselines]
    axes[0].bar(baselines, means, yerr=stds, capsize=5)
    axes[0].set_title('Time to Task Completion (TTC)')
    axes[0].set_ylabel('Simulation Steps (Lower is Better)')
    
    # Plot 2: Safety Violations (NSV)
    means = [avg_metrics['NSV_collisions'][mode]['mean'] for mode in baselines]
    stds = [avg_metrics['NSV_collisions'][mode]['std'] for mode in baselines]
    axes[1].bar(baselines, means, yerr=stds, capsize=5)
    axes[1].set_title('Safety Violations (NSV)')
    axes[1].set_ylabel('Avg. Collisions per Run (Lower is Better)')
    
    # Plot 3: Control Interventions (CI)
    means = [avg_metrics['CI_interventions'][mode]['mean'] * 100 for mode in baselines] # As percentage
    stds = [avg_metrics['CI_interventions'][mode]['std'] * 100 for mode in baselines]
    axes[2].bar(baselines, means, yerr=stds, capsize=5)
    axes[2].set_title('Control Interventions (CI)')
    axes[2].set_ylabel('% of Runs Requiring Intervention (Lower is Better)')
    axes[2].set_ylim(0, 105) # Set Y-limit to 105 to see 100 clearly
    
    plt.tight_layout()
    plt.show()

def plot_time_series(log):
    """
    (FOR SECTION V.B)
    Generates the time-series line graph to show trust dynamics.
    """
    print("\n--- Generating Plot for Section V.B ---")
    
    steps = [entry['step'] for entry in log]
    t_n = [entry['T_N'] for entry in log]
    t_h = [entry['T_H'] for entry in log]
    c_s = [entry['C_S'] for entry in log]
    
    fig, ax = plt.subplots(figsize=(15, 7))
    
    ax.plot(steps, t_n, label='Negotiated Trust (T_N)', color='black', linewidth=3)
    ax.plot(steps, t_h, label='Human Trust (T_H)', color='red', linestyle='--')
    ax.plot(steps, c_s, label='Swarm Confidence (C_S)', color='blue', linestyle=':')
    
    # --- Shade the event regions ---
    # GPS Fail Event
    ax.axvspan(EVENT_GPS_FAIL_START, SIMULATION_STEPS, 
               color='blue', alpha=0.1, label='GPS Failure Period')
    # False Positive Event
    ax.axvspan(EVENT_FALSE_POSITIVE_START, EVENT_FALSE_POSITIVE_END, 
               color='red', alpha=0.2, label='False Positive Period')
    
    ax.set_title('Analysis of Trust Negotiation Dynamics (DTSC Model)', fontsize=16)
    ax.set_xlabel('Simulation Step')
    ax.set_ylabel('Trust / Confidence Level')
    ax.set_ylim(-0.1, 1.1)
    ax.legend(loc='lower left')
    ax.grid(True)
    
    plt.show()

################################################################################
# 6. MAIN EXECUTION (RUNS THE PROJECT)
################################################################################

if __name__ == "__main__":
    
    # --- Part 1: Generate Data for Section V.A ---
    # Run the simulation many times for each baseline to get stable averages.
    
    all_results = defaultdict(list)
    baselines_to_run = ['DTSC', 'Full_Auto', 'Teleop']
    
    print("--- Starting Simulation Runs (for Section V.A) ---")
    print(f"Running {N_SIMULATION_RUNS} trials for each of {len(baselines_to_run)} baselines...")
    start_time = time.time()
    
    for mode in baselines_to_run:
        for i in range(N_SIMULATION_RUNS):
            metrics, _ = run_simulation(baseline_mode=mode)
            all_results[mode].append(metrics)
        print(f"  > Completed '{mode}' runs.")
        
    end_time = time.time()
    print(f"\nTotal simulation time: {end_time - start_time:.2f} seconds")
    
    # --- Part 2: Generate Plots for Section V.A ---
    plot_bar_charts(all_results)
    
    # --- Part 3: Generate Data & Plot for Section V.B ---
    # Run *one* detailed simulation using your DTSC model to get the line graph.
    
    print("\n--- Running Detailed Log for Section V.B ---")
    _, detailed_log = run_simulation(baseline_mode='DTSC', log_time_series=True)
    
    # --- Part 4: Generate Plot for Section V.B ---
    plot_time_series(detailed_log)
    