import numpy as np   # scientific computing tools 
import nutopy as nt  # indirect methods and homotopy

class ConjugateLocus():
    
    def __init__(self, conjugate_locus):
        self.times  = np.array(conjugate_locus[0])
        self.states = np.array(conjugate_locus[1])
        self.alphas = np.array(conjugate_locus[2])
        
    def get_data(self):
        return (self.times.tolist(), self.states.tolist(), self.alphas.tolist())

class WaveFrontLocus():
    
    def __init__(self, wavefront_locus):
        self.tf     = wavefront_locus[0]
        self.states = np.array(wavefront_locus[1])
        self.alphas = np.array(wavefront_locus[2])
        
    def get_data(self):
        return (self.tf, self.states.tolist(), self.alphas.tolist())

class SphereLocus():
    
    def __init__(self, sphere_locus):
        self.tf     = sphere_locus[0]
        self.states = np.array(sphere_locus[1])
        self.alphas = np.array(sphere_locus[2])
        
    def get_data(self):
        return (self.tf, self.states.tolist(), self.alphas.tolist())

class SplittingLocus():
    
    def __init__(self, splitting_locus):
        self.times  = np.array(splitting_locus[0])
        self.states = np.array(splitting_locus[1])
        self.alphas = np.array(splitting_locus[2])
        
    def get_data(self):
        return (self.times.tolist(), self.states.tolist(), self.alphas.tolist())

# Definition of the 2D geometry problem class
class GeometryProblem2D():
    
    def __init__(self, name, Hamiltonian, metric, initial_time, initial_point, data, 
                 steps_for_geodesics=100, modulo=None):
        
        # ----------------------------------------------------------------------------------
        self.name = name
        self.Hamiltonian = Hamiltonian
        self.metric = metric
        self.initial_time = initial_time
        self.initial_point = initial_point
        self.data = data
        self.steps_for_geodesics = steps_for_geodesics
        self.epsilon = 1.0 # Sphere
        self.modulo  = modulo
        
        # ----------------------------------------------------------------------------------
        # Hamiltonian exponential map and its derivatives
        self.extremal = nt.ocp.Flow(self.Hamiltonian)
        
        # ----------------------------------------------------------------------------------
        # Initial covector parameterization and its derivatives up to order 2
        def covector__(q, α):
            g1, g2 = self.metric(q)
            p0 = np.array([np.sin(α)*np.sqrt(g1), 
                           np.cos(α)*np.sqrt(g2)])
            return p0
        def dcovector__(q, α, dα):
            g1, g2 = self.metric(q)
            dp0 = np.array([ np.cos(α)*np.sqrt(g1)*dα, 
                            -np.sin(α)*np.sqrt(g2)*dα])
            return dp0
        def d2covector__(q, α, dα, d2α):
            g1, g2 = self.metric(q)
            d2p0 = np.array([-np.sin(α)*np.sqrt(g1)*dα*d2α, 
                             -np.cos(α)*np.sqrt(g2)*dα*d2α])
            return d2p0
        self.covector = nt.tools.tensorize(dcovector__, d2covector__, \
                                           tvars=(2,))(covector__)

        # ----------------------------------------------------------------------------------
        # Compute a geodesic parameterized by α0 to time t
        # The initial point is fixed to q0 = initial_point and the 
        # initial time is fixed to t0 = initial_time
        def tspan(t0, t):
            N = self.steps_for_geodesics
            return list(np.linspace(t0, t, N))
        
        def geodesic(t, α0):
            t0     = self.initial_time
            q0     = self.initial_point
            p0     = self.covector(q0, α0)
            q, p   = self.extremal(t0, q0, p0, tspan(t0, t))
            return q
        
        def initial_cotangent_point(α0):
            q0 = self.initial_point
            p0 = self.covector(q0, α0)
            return np.concatenate([q0, p0])
            
        self.geodesic = geodesic
        self.initial_cotangent_point = initial_cotangent_point
    
        # ----------------------------------------------------------------------------------
        # Jacobi field: dz(t, p(α0), dp(α0))
        @nt.tools.vectorize(vvars=(1,))
        def jacobi__(t, α0):
            t0     = self.initial_time
            q0     = self.initial_point
            p0, dp0  = self.covector(q0, (α0, 1.))
            (q, dq), (p, dp) = self.extremal(t0, q0, (p0, dp0), t)
            return (q, dq), (p, dp)
        
        # Derivative of dq w.r.t. t and α0
        def djacobi__(t, α0):
            t0     = self.initial_time
            q0     = self.initial_point
            #
            p0, dp0, d2p0 = self.covector(q0, (α0, 1., 1.))
            #
            (q, dq1, d2q), (p, dp1, _) = self.extremal(t0, q0, \
                                                (p0, dp0, dp0), t)
            (q, dq2), (p, dp2)         = self.extremal(t0, q0, \
                                                  (p0, d2p0), t)
            #
            hv, dhv   = self.Hamiltonian.vec(t, (q, dq1), (p, dp1))
            #
            ddqda     = d2q+dq2  # ddq/dα
            ddqdt     = dhv[0:2] # ddq/dt
            return (q, dq1), (p, dp1), (ddqdt, ddqda)
        
        # Function to compute conjugate time together with 
        # the initial angle and the associated conjugate point
        #
        # conjugate(tc, qc, a0) = ( det( dq(tc, a0), 
        #                                Hv(tc, z(tc, a0)) ), 
        #                        qc - pi_q(z(tc, a0)) ),
        #
        # where pi_q(q, p) = q and 
        # z(t, a) = extremal(t0, q0, p(a), t).
        #
        # Remark: y = (tc, qc)
        #
        def conjugate__(y, a):
            tc = y[0]
            qc = y[1:3]
            α0 = a[0]
            #
            (q, dq), (p, dp) = jacobi__(tc, α0)
            hv     = self.Hamiltonian.vec(tc, q, p)[0:2]
            #
            c      = np.zeros(3)
            c[0]   = np.linalg.det([hv, dq]) / tc
            c[1:3] = qc - q
            return c

        def conjugate(tc, α0):
            (q, dq), (p, dp) = jacobi__(tc, α0)
            hv = self.Hamiltonian.vec(tc, q, p)[0:2]
            return np.linalg.det([hv, dq]) / tc
            
        self.conjugate = conjugate

        # Derivative of conjugate
        def dconjugate__(y, a):
            tc = y[0]
            qc = y[1:3]
            α0 = a[0]
            #
            (q, dq), (p, dp), (ddqdt, ddqda) = djacobi__(tc, α0)
            #
            # dc/da
            hv, dhv     = self.Hamiltonian.vec(tc, (q, dq), (p, dp))
            dcda        = np.zeros((3, 1))
            dnum        = np.linalg.det([dhv[0:2], dq]) + \
            np.linalg.det([hv[0:2], ddqda]) 
            dcda[0,0]   = dnum/tc
            dcda[1:3,0] = -dq
            #
            # dc/dy = (dc/dt, dc/dq)
            hv, dhv     = self.Hamiltonian.vec(tc, (q, hv[0:2]), (p, hv[2:4]))
            dcdy        = np.zeros((3, 3))
            num         = np.linalg.det([hv[0:2], dq])
            dnum        = np.linalg.det([dhv[0:2], dq]) + \
            np.linalg.det([hv[0:2], ddqdt]) 
            dcdy[0,0]   = (dnum*tc-num)/tc**2
            dcdy[1:3,0] = -hv[0:2]
            dcdy[1,1]   = 1.
            dcdy[2,2]   = 1.
            return dcdy, dcda
        
        def conjugate_locus(α0, αf):
        
            t0     = self.initial_time
            q0     = self.initial_point
            
            # -------------------------
            # Get first conjugate point
        
            # Initial guess
            tci    = np.pi
            α      = [α0]
            p0     = self.covector(q0, α[0])
            xi, pi = self.extremal(t0, q0, p0, tci)
        
            yi      = np.zeros(3)
            yi[0]   = tci
            yi[1:3] = xi
        
            # Equations and derivative
            fun   = lambda t: conjugate__(t, α)
            dfun  = lambda t: dconjugate__(t, α)[0]
        
            # Callback
            def print_conjugate_time(infos):
                print('    Conjugate time estimation: \
                tc = %e for α = %e' % (infos.x[0], α[0]), end='\r')
        
            # Options
            opt  = nt.nle.Options(Display='on')
        
            # Conjugate point calculation for 
            # initial homotopic parameter
            print(' > Get first conjugate time and point:\n')
            sol   = nt.nle.solve(fun, yi, df=dfun, \
                                 callback=print_conjugate_time, \
                                 options=opt)
        
            # -------------------
            # Get conjugate locus
        
            # Options
            opt = nt.path.Options(MaxStepSizeHomPar=0.05, \
                                  Display='on');
        
            # Initial solution
            y0 = sol.x
        
            # Callback
            def progress(infos):
                current   = infos.pars[0]-α0
                total     = αf-α0
                barLength = 50
                percent   = float(current * 100.0 / total)
                arrow = '-' * int(percent/100 * barLength - 1) + \
                '>'
                spaces    = ' ' * (barLength - len(arrow))
        
                print('    Progress: [%s%s] %1.2f %%' % \
                      (arrow, spaces, round(percent, 2)), end='\r')
        
            # Conjugate locus calculation
            print('\n\n > Get the conjugate locus for α in \
            [%e, %e]:\n' % (α0, αf))
            sol = nt.path.solve(conjugate__, y0, α0, αf, \
                                options=opt, df=dconjugate__, callback=progress)
            print('\n')
        
            return ConjugateLocus( (sol.xout[:, 0], sol.xout[:, 1:3], sol.parsout) ) # (t, q, α)
        
        self.conjugate_locus = conjugate_locus

        # ----------------------------------------------------------------------------------
        # Equation to calculate wavefronts
        def wavefront_eq__(q, α0, tf):
            t0     = self.initial_time
            q0     = self.initial_point
            p0    = self.covector(q0, α0[0])
            qf, _ = self.extremal(t0, q0, p0, tf)
            return q - qf
        
        # Derivative
        def dwavefront_eq__(q, dq, α0, dα0, tf):
            t0     = self.initial_time
            q0     = self.initial_point
            p0, dp0      = self.covector(q0, (α0[0], dα0[0]))
            (qf, dqf), _ = self.extremal(t0, q0, (p0, dp0), tf)
            return q-qf, dq - dqf
        
        wavefront_eq = nt.tools.tensorize(dwavefront_eq__, \
                                          tvars=(1, 2), \
                                          full=True)(wavefront_eq__)
        
        # Function to compute wavefront at time tf, q0 being fixed
        def wavefront(tf, α0, αf, *, options=None):
        
            #
            t0     = self.initial_time
            q0     = self.initial_point
            
            # Options
            if options is None:
                opt = nt.path.Options(Display='off', MaxStepSizeHomPar=0.01, MaxIterCorrection=10)
            else:
                opt = options
        
            # Initial solution
            p0     = self.covector(q0, α0)
            xf0, _ = self.extremal(t0, q0, p0, tf)
        
            # callback
            def progress(infos):
                current   = infos.pars[0]-α0
                total     = αf-α0
                barLength = 50
                percent   = float(current * 100.0 / total)
                arrow = '-' * int(percent/100.0 * barLength - 1)+ \
                '>'
                spaces    = ' ' * (barLength - len(arrow))
        
                print('    Progress: [%s%s] %1.2f %%' % \
                      (arrow, spaces, round(percent, 2)), end='\r')
        
            # wavefront computation
            print('\n > Get wavefront for tf =', tf, '\n')
            sol = nt.path.solve(wavefront_eq, xf0, α0, αf, \
                                args=tf, options=opt, \
                                df=wavefront_eq, callback=progress)
            print('\n')
        
            return WaveFrontLocus( (tf, sol.xout, sol.parsout) ) # tf, q, α
        
        self.wavefront = wavefront
        
        # ----------------------------------------------------------------------------------
        # Equations to compute Split(q0)
        def split_eq__(y, α2):
            t0     = self.initial_time
            q0     = self.initial_point
            # y = (t, α1, q)
            t     = y[0]
            a1    = y[1]
            q     = y[2:4]
            a2    = α2[0]
            q1, _ = self.extremal(t0, q0, self.covector(q0, a1), t)
            q2, _ = self.extremal(t0, q0, self.covector(q0, a2), t)
            eq    = np.zeros(4)
            if self.modulo is None:
                eq[0:2] = q - q1
                eq[2:4] = q - q2
            else:
                eq[0:2] = q - (q1 % self.modulo)
                eq[2:4] = q - (q2 % self.modulo)
            return eq

        def split_eq(t, a1, a2):
            t0     = self.initial_time
            q0     = self.initial_point
            q1, _ = self.extremal(t0, q0, self.covector(q0, a1), t)
            q2, _ = self.extremal(t0, q0, self.covector(q0, a2), t)
            eq    = np.zeros(2)
            if self.modulo is None:
                eq[0:2] = q1 - q2
            else:
                eq[0:2] = (q1 % self.modulo) - (q2 % self.modulo)
            return eq

        self.split_eq = split_eq
    
        # Derivative
        def dsplit_eq__(y, dy, α2, dα2):
            t0     = self.initial_time
            q0     = self.initial_point
            t, dt   = y[0], dy[0]
            a1, da1 = y[1], dy[1]
            q, dq   = y[2:4], dy[2:4]
            a2, da2 = α2[0], dα2[0]
            (q1, dq1), _ = self.extremal(t0, q0, self.covector(q0, \
                                                     (a1, da1)), \
                                    (t, dt))
            (q2, dq2), _ = self.extremal(t0, q0, self.covector(q0, \
                                                     (a2, da2)), \
                                    (t, dt))
            eq, deq      = np.zeros(4), np.zeros(4)
            if self.modulo is None:
                eq[0:2], deq[0:2] = q-q1, dq-dq1
                eq[2:4], deq[2:4] = q-q2, dq-dq2
            else:
                eq[0:2], deq[0:2] = q - (q1 % self.modulo), dq-dq1
                eq[2:4], deq[2:4] = q - (q2 % self.modulo), dq-dq2
            return eq, deq        
        
        split_eq__ = nt.tools.tensorize(dsplit_eq__, tvars=(1, 2), \
                                full=True)(split_eq__)

        self.split_fun = split_eq__
        
        # Function to compute the splitting locus
        def splitting_locus(q, a1, t, a2, α0, αf, *, options=None):
       
            t0     = self.initial_time
            q0     = self.initial_point 
            
            # Options
            if options is None:
                opt  = nt.path.Options(MaxStepSizeHomPar=0.05, Display='off')
            else:
                opt = options
        
            # Initial solution
            y0 = np.array([t, a1, q[0], q[1]])
            b0 = a2
        
            # callback
            def progress_bis(infos):
                current   = b0-infos.pars[0]
                total     = αf-α0+b0-α0
                barLength = 50
                percent   = float(current * 100.0 / total)
                arrow = '-' * int(percent/100.0 * barLength - 1)+ \
                '>'
                spaces    = ' ' * (barLength - len(arrow))
        
                print('    Progress: [%s%s] %1.2f %%' % \
                      (arrow, spaces, round(percent, 2)), end='\r')
        
            # First homotopy
            print('\n > Get splitting locus\n')
            sol  = nt.path.solve(split_eq__, y0, b0, α0, \
                                 options=opt, df=split_eq__, \
                                 callback=progress_bis)
            ysol = sol.xf
        
            # callback
            def progress(infos):
                current   = b0-α0+infos.pars[0]-α0
                total     = αf-α0+b0-α0
                barLength = 50
                percent   = float(current * 100.0 / total)
                arrow = '-' * int(percent/100.0 * barLength - 1)+ \
                '>'
                spaces    = ' ' * (barLength - len(arrow))
        
                print('    Progress: [%s%s] %1.2f %%' % \
                      (arrow, spaces, round(percent, 2)), end='\r')
        
            # Splitting locus computation
            sol = nt.path.solve(split_eq__, ysol, α0, αf, \
                                options=opt, df=split_eq__, \
                                callback=progress)
            print('\n')
        
            # (t, q, α)
            # αs contains α1 and α2 from xout[:, 1] and parsout
            # αs must be a numpy array of size (N, 2)
            αs = np.zeros((len(sol.xout), 2))
            αs[:, 0] = sol.xout[:, 1]
            αs[:, 1] = sol.parsout
            return SplittingLocus( ( sol.xout[:, 0], sol.xout[:, 2:4], αs ) )
        
        self.splitting_locus = splitting_locus