import random
import itertools
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Change directory to desktop
os.chdir(os.path.expanduser("~/Desktop"))

setup_times_df = pd.read_excel("433 setups.xlsx", index_col=0)
setup_times = setup_times_df.to_dict()

first_setup_times_df = pd.read_excel("Setup Times Full.xlsx", index_col=0, header=None, names=['Job', 'O'])
first_setup_times = first_setup_times_df['O'].to_dict()

processing_times_df = pd.read_excel("Processing Times Full.xlsx", index_col=0)
processing_times = processing_times_df.to_dict()

num_jobs = 433
num_machines = 39

# Genetic Algorithm Parameters
population_size = 100
num_generations = 200
mutation_rate = 0.2
elite_percentage = 0.1
tournament_size = 5
random.seed(42)


def generate_individual():
    return [random.randint(1, num_machines) for _ in range(num_jobs)]


def initial_population():
    return [generate_individual() for _ in range(population_size)]


def makespan(individual, machine_job_sequences=None, debugg=0):
    # Initialize machine_job_sequences to a list of empty lists if it's not provided as an argument.
    # Each sublist corresponds to a machine and will hold the sequence of jobs assigned to that machine.
    if machine_job_sequences is None:
        machine_job_sequences = [[] for _ in range(num_machines)]
        # For each job (and its assigned machine) in the individual:
        # Append the job to the sequence of jobs for the assigned machine.
        for job, machine in enumerate(individual, start=1):
            machine_job_sequences[machine - 1].append(job)

    # Initialize machine_times and last_job_on_machine as lists of zeros/None with length equal to the number of machines.
    # machine_times will hold the total processing and setup time for each machine.
    # last_job_on_machine will hold the last job processed by each machine.
    machine_times = [0] * num_machines
    last_job_on_machine = [None] * num_machines

    # Reset the machine_job_sequences
    machine_job_sequences = [[] for _ in range(num_machines)]

    # For each job (and its assigned machine) in the individual:
    for job, machine in enumerate(individual, start=1):
        # Get the processing time for the job on the assigned machine.
        processing_time = processing_times[machine][job]
        # Assume the setup time is the first setup time for the job.
        setup_time = first_setup_times[job]

        # If the machine has processed a job before:
        if last_job_on_machine[machine - 1] is not None:
            # Get the previous job processed by the machine.
            prev_job = last_job_on_machine[machine - 1]
            # Update the setup time to be the setup time between the previous job and the current job.
            setup_time = setup_times[job][prev_job]

        # Add the processing time and setup time to the total time for the machine.
        machine_times[machine - 1] += processing_time + setup_time
        # Update the last job processed by the machine.
        last_job_on_machine[machine - 1] = job
        # Append the job to the sequence of jobs for the machine.
        machine_job_sequences[machine - 1].append(job)

    # Return the maximum total time across all machines (i.e., the makespan),
    # the sequence of jobs for each machine, and the total time for each machine.
    return max(machine_times), machine_job_sequences, machine_times


def fitness(individual):
    return 1 / makespan(individual)[0]


# Selection: Tournament selection
def tournament_selection(population):
    return min(random.sample(population, tournament_size), key=lambda x: makespan(x)[0])


# Crossover: One-point crossover
def crossover(parent1, parent2):
    crossover_point = random.randint(1, num_jobs - 1)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2


def find_zero_setup_pairs():
    zero_setup_pairs = []
    for job1 in range(1, num_jobs + 1):
        for job2 in range(1, num_jobs + 1):
            if job1 != job2 and setup_times[job1][job2] < 0.001:
                zero_setup_pairs.append((job1, job2))
    return zero_setup_pairs


zero_setup_pairs = find_zero_setup_pairs()


def sequence_aware_mutate(individual):
    job_to_mutate = random.randint(0, num_jobs - 1)
    current_machine = individual[job_to_mutate]

    # Find a new random machine that is different from the current machine
    new_machine = current_machine
    while new_machine == current_machine:
        new_machine = random.randint(1, num_machines)

    # Check if job_to_mutate is part of a zero setup time pair
    for pair in zero_setup_pairs:
        if job_to_mutate + 1 in pair:
            other_job = pair[0] if pair[1] == job_to_mutate + 1 else pair[1]

            # If the other job of the pair is already on the new machine, don't mutate
            if other_job in [i + 1 for i, machine in enumerate(individual) if machine == new_machine]:
                return individual

    # Find the job with the same position as the job_to_mutate on the new machine
    position_on_current_machine = [i for i, machine in enumerate(individual) if machine == current_machine].index(
        job_to_mutate)
    jobs_on_new_machine = [i for i, machine in enumerate(individual) if machine == new_machine]
    if position_on_current_machine < len(jobs_on_new_machine):
        job_to_swap = jobs_on_new_machine[position_on_current_machine]
        individual[job_to_mutate] = new_machine
        individual[job_to_swap] = current_machine
    else:
        individual[job_to_mutate] = new_machine

    return individual


def minimize_makespan(machine_job_sequences, machine_times):
    improving = True
    while improving:  # Continue until no more improving moves can be found
        improving = False
        # Find the machine with maximum completion time and the machine with minimum completion time
        max_time_machine = machine_times.index(max(machine_times)) + 1
        print(max_time_machine)
        min_time_machine = machine_times.index(min(machine_times)) + 1
        print(min_time_machine)
        # Only consider the last job in the machine with the maximum completion time
        n_jobs = len(machine_job_sequences[max_time_machine-1])
        for uu in range(n_jobs-1,-1,-1):
            job_to_move = machine_job_sequences[max_time_machine - 1][uu]
            if improving:
                break

            #job_to_move = machine_job_sequences[max_time_machine - 1][-1]
            print(job_to_move)
            # Temporarily move the job to the machine with minimum completion time
            machine_job_sequences[max_time_machine - 1].remove(job_to_move)
            machine_job_sequences[min_time_machine - 1].append(job_to_move)

            individual = [0] * num_jobs

            for machine in range(num_machines):
                for job in machine_job_sequences[machine]:
                    individual[job - 1] = machine + 1  # as job is 1-based and machine is 0-based

            print(max(machine_times))
            new_makespan, _, new_machine_times = makespan(individual, machine_job_sequences)
            print(new_makespan)

            # If the new makespan is less than the old makespan, make the move permanent
            if new_makespan < max(machine_times):
                machine_times = new_machine_times
                improving = True  # A job was moved, so continue to the next iteration
            else:
                # Undo the move
                machine_job_sequences[max_time_machine - 1].append(job_to_move)
                machine_job_sequences[min_time_machine - 1].remove(job_to_move)

    return machine_job_sequences, machine_times


def genetic_algorithm():
    population = initial_population()

    for generation in range(num_generations):
        new_population = []
        elites = sorted(population, key=lambda x: makespan(x)[0])[:int(elite_percentage * population_size)]

        for _ in range(population_size // 2 - len(elites) // 2):
            parent1 = tournament_selection(population)
            parent2 = tournament_selection(population)

            child1, child2 = crossover(parent1, parent2)

            if random.random() < mutation_rate:
                child1 = sequence_aware_mutate(child1)
            if random.random() < mutation_rate:
                child2 = sequence_aware_mutate(child2)

            new_population += [child1, child2]

        population = elites + new_population

    return min(population, key=lambda x: makespan(x)[0]), makespan(min(population, key=lambda x: makespan(x)[0]))[1]


best_solution, machine_job_sequences = genetic_algorithm()
minimized_machine_job_sequences, minimized_machine_times = minimize_makespan(machine_job_sequences, makespan(best_solution, machine_job_sequences)[2])
new_chromosome = [0] * num_jobs
for machine in range(num_machines):
    for job in minimized_machine_job_sequences[machine]:
        new_chromosome[job - 1] = machine + 1

minimized_makespan, _, _ = makespan(new_chromosome, minimized_machine_job_sequences)


# Output for the minimized version
print("\nMinimized makespan:", minimized_makespan)
print(minimized_machine_times)
print("Minimized job orders on the machines:")
for i, sequence in enumerate(minimized_machine_job_sequences, start=1):
    print(f"Machine {i}: {sequence}")

# Update the start times for the new job sequences
minimized_start_times = {}
for machine_id, jobs in enumerate(minimized_machine_job_sequences, start=1):
    machine_start_time = 0
    for job in jobs:
        setup_time = first_setup_times[job]
        processing_time = processing_times[machine_id][job]

        if minimized_machine_job_sequences[machine_id - 1].index(job) > 0:
            prev_job = minimized_machine_job_sequences[machine_id - 1][minimized_machine_job_sequences[machine_id - 1].index(job) - 1]
            setup_time = setup_times[prev_job][job]

        minimized_start_times[(job, machine_id)] = machine_start_time + setup_time
        machine_start_time += setup_time + processing_time


# Plot the Gantt chart for Genetic Algorithm
colors = plt.get_cmap('tab10').colors
if minimized_makespan > 0:
    fig, ax = plt.subplots()
    machines = []

    for machine_id, jobs in enumerate(minimized_machine_job_sequences, start=1):
        machine_name = f"Machine {machine_id}"
        for job in jobs:
            start_time = minimized_start_times[(job, machine_id)]
            processing_time = processing_times[machine_id][job]
            ax.broken_barh([(start_time, processing_time)], (machine_id * 10, 9),
                           facecolors=(colors[job % len(colors)]))
            machines.append(machine_name)
            ax.text(start_time + processing_time / 2, machine_id * 10 + 4, f"Job{job}", ha='center', va='center',
                    color='black', fontsize=6)

    plt.title("Gantt chart (Minimized Makespan)")
    plt.xlabel("Time")
    plt.yticks([k * 10 + 5 for k in range(1, num_machines + 1)],
               [f"Machine {k}" for k in range(1, num_machines + 1)])
    plt.grid(axis='x')
    ax.set_xlim(left=0)

    plt.show()


completion_times = {}

for machine_id, jobs in enumerate(machine_job_sequences, start=1):
    machine_start_time = 0
    for job in jobs:
        setup_time = first_setup_times[job]
        processing_time = processing_times[machine_id][job]

        if machine_job_sequences[machine_id - 1].index(job) > 0:
            prev_job = machine_job_sequences[machine_id - 1][machine_job_sequences[machine_id - 1].index(job) - 1]
            setup_time = setup_times[prev_job][job]

        completion_time = machine_start_time + setup_time + processing_time
        machine_start_time += setup_time + processing_time
        completion_times[(machine_id, job)] = completion_time
df_jobs_list = []

for machine_id, jobs in enumerate(machine_job_sequences, start=1):
    job_list = [job for job in jobs]
    completion_time_list = [f"Completion time: {completion_times[(machine_id, job)]:.2f}" for job in jobs]

    df_temp = pd.DataFrame({f"Machine_{machine_id}": job_list, f"Completion_time_{machine_id}": completion_time_list})
    df_jobs_list.append(df_temp)

df_jobs = pd.concat(df_jobs_list, axis=0)

df_jobs.to_excel("Copy of Output Excel.xlsx", index=False)
