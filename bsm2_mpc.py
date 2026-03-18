# bsm2_mpc.py
import numpy as np
import casadi as ca
import do_mpc

class BSM2_LCA_MPC:
    def __init__(self):
        # 1. Initialize Model
        self.model = do_mpc.model.Model('continuous')
        
        # Define States (x) - e.g., concentrations in Aerobic Reactor 4
        self.S_NH_4 = self.model.set_variable(var_type='_x', var_name='S_NH_4')
        self.S_NO_4 = self.model.set_variable(var_type='_x', var_name='S_NO_4')
        self.S_O_4  = self.model.set_variable(var_type='_x', var_name='S_O_4')
        self.F_to_M = self.model.set_variable(var_type='_x', var_name='F_to_M') # From perf_risk
        
        # Define Manipulated Variables (u)
        self.KLa4   = self.model.set_variable(var_type='_u', var_name='KLa4')
        self.Q_intr = self.model.set_variable(var_type='_u', var_name='Q_intr')
        
        # (Define RHS / ODEs for your internal reduced ASM1 model here)
        # self.model.set_rhs('S_O_4', ... ) 
        self.model.setup()
        
        # 2. Initialize Controller
        self.mpc = do_mpc.controller.MPC(self.model)
        setup_mpc = {
            'n_horizon': 10,       # Prediction horizon
            't_step': 0.0104,      # 15 minutes in days (BSM2 time unit)
            'n_robust': 0,
            'store_full_solution': False,
        }
        self.mpc.set_param(**setup_mpc)
        
        # 3. Objective Function (LCA Minimization)
        # Simplified: Minimize Aeration Energy (KLa) and Pumping (Q_intr)
        lterm = 1.0 * self.KLa4 + 0.004 * self.Q_intr 
        mterm = 1.0 * self.S_NH_4  # Penalize remaining Ammonia at the end of horizon
        self.mpc.set_objective(mterm=mterm, lterm=lterm)
        self.mpc.set_rterm(KLa4=1e-2, Q_intr=1e-4) # Move suppression
        
        # 4. SUPERVISORY CONSTRAINTS (Risk Aversion)
        # Based on perf_risk_bsm2.m logic to prevent Settling Problems:
        
        # A. Prevent Low DO Bulking (Aerobic Bulking)
        # Fuzzy logic triggers high risk when DO is low and F/M is high.
        # Hard constraint: Never let S_O_4 drop below 1.0 mg/L.
        self.mpc.bounds['lower', '_x', 'S_O_4'] = 1.0
        
        # B. Prevent N-Deficiency Bulking
        # Triggers when Effluent Ammonia/Nitrate gets too low causing nutrient starvation.
        # Constraint: Ensure a minimum baseline of N in the system.
        self.mpc.bounds['lower', '_x', 'S_NH_4'] = 0.5 
        
        # C. Actuator Physical Limits (from sensorinit_bsm2.m)
        self.mpc.bounds['lower', '_u', 'KLa4'] = 0.0
        self.mpc.bounds['upper', '_u', 'KLa4'] = 360.0 # Max aeration
        
        self.mpc.bounds['lower', '_u', 'Q_intr'] = 0.0
        self.mpc.bounds['upper', '_u', 'Q_intr'] = 103240.0 # Max internal recycle (5 * Qin0)
        
        self.mpc.setup()

    def get_action(self, current_states):
        # Format input from MATLAB into numpy array
        x0 = np.array(current_states).reshape(-1, 1)
        # Solve MPC for this step
        u0 = self.mpc.make_step(x0)
        # Return control actions to MATLAB as a list
        return [float(u0[0]), float(u0[1])]

# Global instance to keep the MPC memory alive between MATLAB calls
mpc_instance = BSM2_LCA_MPC()

def step_mpc(S_NH_4, S_NO_4, S_O_4, F_to_M):
    states = [S_NH_4, S_NO_4, S_O_4, F_to_M]
    return mpc_instance.get_action(states)