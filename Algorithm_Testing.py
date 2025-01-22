import pandas as pd
import os
# Change directory to desktop
os.chdir(os.path.expanduser("~/Desktop"))
from gurobipy import *
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# Creating a new model
model = gp.Model()
# Define indices
M = range(1, 4)  # Set of machines
N = range(1, 6)  # Set of jobs
O = [0]           # Dummy job
a = range(1,6)

# Define parameters
bigM = 100000  # Big number
P = pd.read_excel("P_Table_3_5.xlsx", index_col=0)
S = pd.read_excel("S_Table_3_5.xlsx", index_col=0)
S_O = pd.read_excel("S_0_Table_3_5.xlsx", index_col=0)
#due_dates = pd.read_excel("duedateexample.xlsx", index_col=0)
lambda_ = 0

# Define variables
x = model.addVars(N, vtype=GRB.CONTINUOUS, name="starting time of job" ) #starting time of job i
C_max = model.addVar(vtype=GRB.CONTINUOUS, name="Makespan (C_max)") #makespan
C = model.addVars(N, vtype=GRB.CONTINUOUS, name="Completion time of job") #completion time of job j
y = model.addVars(N, M, vtype=GRB.BINARY, name="if job i is assigned to machine k= ") #1 if job i is assigned to machine k or 0 o.w.
z = model.addVars(N, N, M, vtype=GRB.BINARY, name="if job j is processed right after job i on machine k= ") #1 if job j is processed right after job i on machine k or 0 o.w.
z_O = model.addVars(N, M, vtype=GRB.BINARY, name="if job i is processed as the first job on machine k= ") #1 if job i is processed as the first job on machine k or 0 o.w.
e_plus = model.addVars(a,vtype=GRB.CONTINUOUS, name='Delay_minutes')
e_minus = model.addVars(a, vtype=GRB.CONTINUOUS, name='Early_minutes')

# Objective function
model.setObjective(C_max + lambda_ * quicksum(e_plus[i] for i in N ), GRB.MINIMIZE)


# Constraints
for i in N:
    model.addConstr(x[i] + gp.quicksum((S_O.loc[i,1] * z_O[i, k] for k in M))
                    + gp.quicksum(S.loc[h, i] * z[h, i, k] for h in N if h != i for k in M)
                    + gp.quicksum(P.loc[i, k] * y[i, k] for k in M) == C[i])

for i in N:
    model.addConstr(C[i] <= C_max)
    model.addConstr(C[i] >= 0)
    model.addConstr(x[i] >= 0)


for i in N:
    for j in N:
        if i != j:
            model.addConstr(C[i] <= x[j] + bigM * (1 - gp.quicksum(z[i, j, k] for k in M)))


for i in N:
    model.addConstr(gp.quicksum(y[i, k] for k in M) == 1)

for k in M:
    model.addConstr(gp.quicksum(z_O[j, k] for j in N) == 1)

for i in N:
    for k in M:
        model.addConstr(gp.quicksum(z[j, i, k] for j in N if i != j) + z_O[i, k] == y[i, k])

for i in N:
    for k in M:
        for j in N:
            if i != j:
                model.addConstr((z[i, j, k]) <= y[i, k])
                model.addConstr((z[i, j, k]) <= y[j, k])

for i in N:
    for k in M:
        model.addConstr(gp.quicksum(z[i, j, k] for j in N if i != j) <= 1)
        model.addConstr(gp.quicksum(z[j, i, k] for j in N if i != j) <= 1)

for i in N:
    model.addConstr(C[i] - due_dates.loc[i,1] == e_plus[i] - e_minus[i])
    model.addConstr(e_plus[i] >= 0)
    model.addConstr(e_minus[i] >= 0)
# Optimize the model
model.optimize()

# Print the solution
##if model.status == GRB.OPTIMAL:
    #print(f"Optimal objective value: {model.objVal}")
    #for v in model.getVars():
       # if v.x >= 1e-6:
            #print(f"{v.varName}: {round(v.x,5)}")
#else:
    #print("Model is infeasible.")

if model.status == GRB.OPTIMAL:
    print(f"Optimal objective value: {model.objVal}\n")
    print("Variable values:")
    for v in model.getVars():

        if v.x >= 1e-6:
            variable_name = v.varName
            if "starting time of job" in variable_name:
                job_number = variable_name.split()[-1]
                print(f"Starting time of job {job_number}: {round(v.x, 5)}")
            if "Completion time of job" in variable_name:
                job_number = variable_name.split()[-1]
                print(f"Completion time of job {job_number}: {round(v.x, 5)}")
            if "if job i is assigned to machine k= " in variable_name:
                job_number, machine_number = variable_name.split("[")[1].split("]")[0].split(",")
                print(f"Job {job_number} assigned to machine {machine_number}")
            if "if job j is processed right after job i on machine k= " in variable_name:
                job_number_i, job_number_j, machine_number = variable_name.split("[")[1].split("]")[0].split(",")
                print(f"Job {job_number_j} processed right after job {job_number_i} on machine {machine_number}")
            if "if job i is processed as the first job on machine k= " in variable_name:
                job_number, machine_number = variable_name.split("[")[1].split("]")[0].split(",")
                print(f"Job {job_number} processed as the first job on machine {machine_number}")
            if "Delay_minutes" in variable_name:
                job_number = variable_name.split("[")[1].split("]")[0].split(",")
                if v.x > 1e-6:
                    print(f"Job {job_number} is delayed with : {round(v.x, 5)} minutes")
            if "Early_minutes" in variable_name:
                job_number = variable_name.split("[")[1].split("]")[0].split(",")
                if v.x > 1e-6:
                    print(f"Job {job_number} is early with : {round(v.x, 5)} minutes")

    print("\n")
else:
    print("Model is infeasible.")

# Add a dictionary to store the starting time of each job on each machine
start_times = {}

# Calculate the start time for each job on the assigned machine considering the setup time
for k in M:
    for i in N:
        if y[i, k].x > 0.5:
            if z_O[i, k].x > 0.5:
                start_time = x[i].x + S_O.loc[i, 1]
            else:
                prev_job = [j for j in N if z[j, i, k].x > 0.5][0]
                start_time = x[i].x + S.loc[prev_job, i]
            start_times[(i, k)] = start_time



# Plot the Gantt chart
colors = plt.get_cmap('tab10').colors  # Use a colormap to support more colors
if model.status == GRB.OPTIMAL:
    fig, ax = plt.subplots()
    machines = []

    # Modify the Gantt chart plot code to use the start times calculated above
    for k in M:
        for i in N:
            if y[i, k].x > 0.5:
                machine_name = f"Machine {k}"
                start_time = start_times[(i, k)]
                processing_time = P.loc[i, k]
                ax.broken_barh([(start_time, processing_time)], (k * 10, 9),
                               facecolors=(colors[i % len(colors)]))  # Use colors in a cyclic way
                machines.append(machine_name)
                ax.text(start_time + processing_time / 2, k * 10 + 4, f"Job {i}", ha='center', va='center', color='black')

    # (existing Gantt chart plotting code)

    plt.title("Gantt chart")
    plt.xlabel("Time")
    plt.yticks([k * 10 + 5 for k in M], [f"Machine {k}" for k in M])  # Set yticks based on the number of machines
    plt.grid(axis='x')
    ax.set_xlim(left=0)

    plt.show()


