import numpy as np
from scipy.optimize import fsolve, root
import matplotlib.pyplot as plt

def f1(x):
    """
    First function: -1/(100-x^2)^0.5
    Domain: |x| < 10 (to avoid division by zero)
    """
    return -1.0 / np.sqrt(100.0 - x**2)

def f2(x, a, b):
    """
    Second function: (a*b/2)(Exp[x] - Exp[-x])(100 - x^2)
    """
    return (a * b / 2.0) * (np.exp(x) - np.exp(-x)) * (100.0 - x**2)

def f3(x, a, b):
    """
    Third function: 0.5*a*(Exp[b*x] + Exp[-b*x]) - 6.7
    """
    return 0.5 * a * (np.exp(b * x) + np.exp(-b * x)) - 6.7

def df1_dx(x):
    """
    Derivative of f1 with respect to x
    """
    return -x / ((100.0 - x**2)**1.5)

def df2_dx(x, a, b):
    """
    Derivative of f2 with respect to x
    """
    exp_x = np.exp(x)
    exp_negx = np.exp(-x)
    sinh_x = exp_x - exp_negx
    cosh_x = exp_x + exp_negx
    return (a * b / 2.0) * (cosh_x * (100.0 - x**2) - sinh_x * 2.0 * x)

def df3_dx(x, a, b):
    """
    Derivative of f3 with respect to x
    """
    return 0.5 * a * b * (np.exp(b * x) - np.exp(-b * x))

def system_equations(vars):
    """
    System of equations to solve:
    - f1(x) = f2(x, a, b)  [equality 1]
    - f2(x, a, b) = f3(x, a, b)  [equality 2]
    - df1/dx = df2/dx  [gradient equality 1]
    - df2/dx = df3/dx  [gradient equality 2]
    
    Parameters:
    -----------
    vars : array-like
        [x, a, b]
    
    Returns:
    --------
    residuals : array
        [f1-f2, f2-f3, df1/dx - df2/dx, df2/dx - df3/dx]
    """
    x, a, b = vars
    
    # Check domain constraint for f1
    if abs(x) >= 10.0:
        return [1e10, 1e10, 1e10, 1e10]
    
    # Calculate function values
    val1 = f1(x)
    val2 = f2(x, a, b)
    val3 = f3(x, a, b)
    
    # Calculate derivatives
    dval1 = df1_dx(x)
    dval2 = df2_dx(x, a, b)
    dval3 = df3_dx(x, a, b)
    
    # Return residuals
    return [
        val1 - val2,      # f1 = f2
        val2 - val3,      # f2 = f3
        dval1 - dval2,    # df1/dx = df2/dx
        dval2 - dval3     # df2/dx = df3/dx
    ]

def solve_system(initial_guess=None, method='auto', verbose=True):
    """
    Solve the system of equations to find x, a, b.
    
    Parameters:
    -----------
    initial_guess : array-like, optional
        Initial guess [x, a, b]. If None, uses multiple random guesses.
    method : str
        Method for scipy.optimize.root. Options: 'hybr', 'lm', 'broyden1', etc.
    verbose : bool
        If True, print diagnostic information
    
    Returns:
    --------
    solutions : list
        List of solution dictionaries with keys: x, a, b, success, message
    """
    if initial_guess is None:
        # Try multiple initial guesses with focused range
        guesses = []
        # Focus on smaller x values (near zero) where functions are more likely to intersect
        # Exclude b values too close to zero
        for x_val in np.linspace(-5.0, 5.0, 11):  # 11 points from -5 to 5
            for a_val in [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:  # Wider range of a
                for b_val in [0.1, 0.5, 1.0, 2.0, 5.0, -0.1, -0.5, -1.0, -2.0, -5.0]:  # b must be significant (not near 0)
                    guesses.append([x_val, a_val, b_val])
        
        # Additional strategic guesses focusing on promising regions
        guesses.extend([
            # Near zero x values with various a, b combinations
            [0.0, 0.01, 0.01], [0.0, 0.1, 0.1], [0.0, 1.0, 1.0], 
            [0.0, 0.5, 0.5], [0.0, 2.0, 1.0], [0.0, 1.0, 2.0],
            [0.0, 0.1, 1.0], [0.0, 1.0, 0.1], [0.0, 5.0, 1.0],
            
            # Small positive x
            [0.5, 0.1, 0.1], [0.5, 1.0, 1.0], [0.5, 0.5, 0.5],
            [1.0, 0.1, 0.1], [1.0, 1.0, 1.0], [1.0, 0.5, 0.5],
            [1.0, 0.01, 1.0], [1.0, 1.0, 0.01], [1.0, 2.0, 1.0],
            
            # Small negative x
            [-0.5, 0.1, 0.1], [-0.5, 1.0, 1.0], [-0.5, 0.5, 0.5],
            [-1.0, 0.1, 0.1], [-1.0, 1.0, 1.0], [-1.0, 0.5, 0.5],
            [-1.0, 0.01, 1.0], [-1.0, 1.0, 0.01], [-1.0, 2.0, 1.0],
            
            # Medium x values with smaller a, b
            [2.0, 0.01, 0.1], [2.0, 0.1, 0.1], [2.0, 0.5, 0.5],
            [-2.0, 0.01, 0.1], [-2.0, 0.1, 0.1], [-2.0, 0.5, 0.5],
            [3.0, 0.01, 0.1], [3.0, 0.1, 0.5], [3.0, 0.5, 0.1],
            [-3.0, 0.01, 0.1], [-3.0, 0.1, 0.5], [-3.0, 0.5, 0.1],
            
            # Larger x values with very small parameters
            [4.0, 0.001, 0.01], [4.0, 0.01, 0.1], [4.0, 0.1, 0.1],
            [-4.0, 0.001, 0.01], [-4.0, 0.01, 0.1], [-4.0, 0.1, 0.1],
            [5.0, 0.001, 0.01], [5.0, 0.01, 0.1], [5.0, 0.1, 0.1],
            [-5.0, 0.001, 0.01], [-5.0, 0.01, 0.1], [-5.0, 0.1, 0.1],
        ])
    else:
        guesses = [initial_guess]
    
    solutions = []
    seen = set()
    
    # Only use 'lm' method for overdetermined systems (4 equations, 3 unknowns)
    if method == 'auto':
        methods_to_try = ['lm']  # Levenberg-Marquardt handles overdetermined systems
    else:
        methods_to_try = [method] if method == 'lm' else ['lm']  # Force 'lm' for overdetermined
    
    for guess in guesses:
        # Ensure x is in valid domain
        if abs(guess[0]) >= 10.0:
            guess[0] = np.sign(guess[0]) * 9.9
        
        for meth in methods_to_try:
            try:
                if meth == 'lm':
                    result = root(system_equations, guess, method=meth, options={'xtol': 1e-10, 'ftol': 1e-10})
                else:
                    result = root(system_equations, guess, method=meth, options={'xtol': 1e-10})
                
                x, a, b = result.x
                
                # Check if solution is valid
                if abs(x) >= 10.0:
                    continue
                
                # Check if solution satisfies equations
                residuals = system_equations([x, a, b])
                max_residual = max(abs(r) for r in residuals)
                
                # Round to avoid duplicate solutions due to numerical precision
                x_rounded = round(x, 6)
                a_rounded = round(a, 6)
                b_rounded = round(b, 6)
                key = (x_rounded, a_rounded, b_rounded)
                
                # Accept solutions with relaxed tolerance (accept close solutions)
                # Don't filter by b here - we'll filter at the end
                if key not in seen and max_residual < 0.1:
                    seen.add(key)
                    solutions.append({
                        'x': x,
                        'a': a,
                        'b': b,
                        'success': result.success,
                        'message': result.message,
                        'max_residual': max_residual,
                        'residuals': residuals,
                        'method': meth
                    })
                    if verbose:
                        print(f"Found solution with method {meth}: x={x:.6f}, a={a:.6f}, b={b:.6f}, residual={max_residual:.2e}")
                    break  # Found solution, try next guess
            except Exception as e:
                if verbose:
                    print(f"Error with guess {guess} using method {meth}: {e}")
                continue
    
    return solutions

def verify_solution(x, a, b, tol=1e-6):
    """
    Verify that a solution satisfies all conditions.
    
    Parameters:
    -----------
    x, a, b : float
        Solution values
    tol : float
        Tolerance for equality checks
    
    Returns:
    --------
    dict with verification results
    """
    if abs(x) >= 10.0:
        return {'valid': False, 'reason': 'x out of domain'}
    
    val1 = f1(x)
    val2 = f2(x, a, b)
    val3 = f3(x, a, b)
    
    dval1 = df1_dx(x)
    dval2 = df2_dx(x, a, b)
    dval3 = df3_dx(x, a, b)
    
    eq1_ok = abs(val1 - val2) < tol
    eq2_ok = abs(val2 - val3) < tol
    grad1_ok = abs(dval1 - dval2) < tol
    grad2_ok = abs(dval2 - dval3) < tol
    
    return {
        'valid': eq1_ok and eq2_ok and grad1_ok and grad2_ok,
        'f1': val1,
        'f2': val2,
        'f3': val3,
        'df1_dx': dval1,
        'df2_dx': dval2,
        'df3_dx': dval3,
        'eq1_error': abs(val1 - val2),
        'eq2_error': abs(val2 - val3),
        'grad1_error': abs(dval1 - dval2),
        'grad2_error': abs(dval2 - dval3)
    }

def plot_functions(x_range, a, b, solutions=None):
    """
    Plot the three functions and mark intersection points.
    
    Parameters:
    -----------
    x_range : tuple
        (x_min, x_max) range for plotting
    a, b : float
        Parameter values for f2 and f3
    solutions : list, optional
        List of solution dictionaries to mark on plot
    """
    x_vals = np.linspace(x_range[0], x_range[1], 1000)
    
    # Filter x values to valid domain for f1
    x_valid = x_vals[np.abs(x_vals) < 10.0]
    
    y1 = f1(x_valid)
    y2 = f2(x_valid, a, b)
    y3 = f3(x_valid, a, b)
    
    plt.figure(figsize=(12, 8))
    
    plt.plot(x_valid, y1, label='f1(x) = -1/√(100-x²)', linewidth=2)
    plt.plot(x_valid, y2, label=f'f2(x) = (a*b/2)(e^x - e^(-x))(100-x²) [a={a:.4f}, b={b:.4f}]', linewidth=2)
    plt.plot(x_valid, y3, label=f'f3(x) = 0.5*a*(e^(bx) + e^(-bx)) - 6.7 [a={a:.4f}, b={b:.4f}]', linewidth=2)
    
    # Mark solutions
    if solutions:
        for sol in solutions:
            x_sol = sol['x']
            if abs(x_sol) < 10.0:
                y_sol = f1(x_sol)
                plt.plot(x_sol, y_sol, 'ro', markersize=10, label=f"Solution: x={x_sol:.4f}, a={sol['a']:.4f}, b={sol['b']:.4f}")
    
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.title('Functions f1, f2, and f3', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    plt.axvline(x=0, color='k', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("Solving system of equations:")
    print("  f1(x) = -1/sqrt(100-x^2)")
    print("  f2(x, a, b) = (a*b/2)(e^x - e^(-x))(100 - x^2)")
    print("  f3(x, a, b) = 0.5*a*(e^(b*x) + e^(-b*x)) - 6.7")
    print("\nConditions:")
    print("  f1(x) = f2(x, a, b) = f3(x, a, b)")
    print("  df1/dx = df2/dx = df3/dx")
    print("\n" + "="*60)
    print("Searching for solutions (this may take a moment)...")
    
    # Solve the system with auto method selection
    print("Trying systematic search...")
    solutions = solve_system(method='auto', verbose=False)
    
    if solutions:
        # Group similar solutions together (within tolerance)
        unique_solutions = []
        for sol in solutions:
            is_unique = True
            for existing in unique_solutions:
                # Check if this solution is similar to an existing one
                if (abs(sol['x'] - existing['x']) < 0.1 and 
                    abs(sol['a'] - existing['a']) < 0.1 and 
                    abs(sol['b'] - existing['b']) < 0.001):
                    is_unique = False
                    break
            if is_unique:
                unique_solutions.append(sol)
        
        # Filter out solutions where b is too close to zero (use a more reasonable threshold)
        filtered_solutions = [sol for sol in unique_solutions if abs(sol['b']) > 0.01]
        
        print(f"\nFound {len(solutions)} total solutions, {len(unique_solutions)} unique solutions")
        print(f"After filtering b != 0: {len(filtered_solutions)} solutions remain\n")
        
        if not filtered_solutions:
            print("No solutions found with b != 0. All solutions had b approximately 0.")
            print("This suggests the system may only have solutions when b is very close to zero.")
            print("\nShowing all unique solutions for reference:\n")
            filtered_solutions = unique_solutions
        
        for i, sol in enumerate(filtered_solutions, 1):
            print(f"Solution {i}:")
            print(f"  x = {sol['x']:.6f}")
            print(f"  a = {sol['a']:.6f}")
            print(f"  b = {sol['b']:.6f}")
            print(f"  Max residual: {sol['max_residual']:.2e}")
            
            # Verify solution
            verification = verify_solution(sol['x'], sol['a'], sol['b'])
            print(f"  Function values: f1={verification['f1']:.6f}, f2={verification['f2']:.6f}, f3={verification['f3']:.6f}")
            print(f"  Derivatives: df1/dx={verification['df1_dx']:.6f}, df2/dx={verification['df2_dx']:.6f}, df3/dx={verification['df3_dx']:.6f}")
            print()
        
        # Plot functions with first solution
        if solutions:
            sol = solutions[0]
            x_range = (-9.5, 9.5)
            print(f"Plotting functions with a={sol['a']:.4f}, b={sol['b']:.4f}...")
            plot_functions(x_range, sol['a'], sol['b'], solutions)
    else:
        print("\nNo solutions found. Try different initial guesses or check the equations.")
        print("\nYou can try:")
        print("  solutions = solve_system(initial_guess=[x0, a0, b0])")
        print("  with your own initial guess values.")
        print("\n" + "="*60)
        print("HOW TO USE THIS SCRIPT:")
        print("="*60)
        print("1. Run directly from command line:")
        print("   python utile.py")
        print("\n2. Use in Python interactively:")
        print("   >>> from utile import solve_system, verify_solution")
        print("   >>> solutions = solve_system(initial_guess=[1.0, 1.0, 1.0])")
        print("   >>> if solutions:")
        print("   ...     sol = solutions[0]")
        print("   ...     print(f'x={sol[\"x\"]}, a={sol[\"a\"]}, b={sol[\"b\"]}')")
        print("\n3. Try with different initial guesses:")
        print("   >>> solutions = solve_system(initial_guess=[0.0, 0.1, 0.1])")
        print("   >>> solutions = solve_system(initial_guess=[2.0, 0.5, 0.5])")

