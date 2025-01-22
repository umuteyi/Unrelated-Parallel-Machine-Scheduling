# Unrelated Parallel Machine Scheduling with Squence Dependent Setup Times


## **Overview**

This project addresses the **Unrelated Parallel Machine Scheduling Problem** with sequence-dependent setup times, implemented for **EFESAN Machining**, a company specializing in automotive part production. The objective is to optimize production by minimizing makespan and total tardiness across 39 unrelated parallel machines processing 433 jobs. 

The study involves the development of a **Mixed Integer Linear Programming (MILP)** model and a **Genetic Algorithm (GA)** to solve the problem, improving production efficiency while addressing constraints like sequence-dependent setup times and preemptive scheduling.

---

## **Features**

- **Scheduling Optimization**:
  - Developed a **MILP model** for small-scale problem instances.
  - Designed and implemented a **Genetic Algorithm** to handle large-scale scheduling.

- **Key Metrics**:
  - **Makespan**: The time required to complete all jobs.
  - **Total Tardiness**: The delay in completing jobs beyond their due dates.

- **Advanced Techniques**:
  - **GA Operations**:
    - Population initialization, fitness evaluation, tournament selection, one-point crossover, and sequence-aware mutation.
  - Visualization tools including Gantt charts for comparative analysis of MILP and GA results.

---

## **Dataset**

- **Job Data**:
  - 433 jobs with sequence-dependent setup times and varying processing times.
- **Machine Configuration**:
  - 39 unrelated parallel machines.
- **Performance Data**:
  - Collected real-world production data including setup times, due dates, and order lists.

---

## **Methodology**

1. **Problem Modeling**:
   - Defined the problem using three-field notation.
   - Developed constraints for job scheduling, machine allocation, and tardiness.

2. **MILP Implementation**:
   - Built and tested the model, successfully solving small problem instances.
   - Identified scalability issues for large datasets.

3. **Genetic Algorithm Implementation**:
   - Tuned parameters such as population size, mutation rate, and elite percentage.
   - Enhanced solutions with local search methods.

4. **Evaluation**:
   - Compared MILP and GA results for runtime, solution quality, and scalability.
   - Generated Gantt charts for scheduling insights.

---

## **Results**

- **Efficiency Improvements**:
  - Achieved significant reductions in makespan and tardiness using GA.
- **Scalability**:
  - Solved large-scale scheduling problems beyond the capacity of MILP.



