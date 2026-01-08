import time  
import sympy as sp
def f(x):
    return x**5 - 9*x**4 - x**3 + 17*x**2 - 8*x - 8

x_sym = sp.symbols('x')

expr = x_sym**5 - 9*x_sym**4 - x_sym**3 + 17*x_sym**2 - 8*x_sym - 8

f_prime_expr = sp.diff(expr, x_sym)

f_prime = sp.lambdify(x_sym, f_prime_expr)

def newton(a, tolerance):

    if f(a) == 0: return a, 0.0, 0

    x = a 
    step = 1
    start_time = time.time() 

    while True:
        fx = f(x)
        dfx = f_prime(x)

        if dfx == 0: 
            print("❌ 기울기가 0이라 계산 불가!")
            return None

        x_new = x - fx / dfx

        print(f"Step {step}: 현재값={x:.8f}, 다음값={x_new:.8f}")

        if abs(x_new - x) < tolerance:
            break

        x = x_new 
        step += 1

        if step > 100: 
            print("⚠️ 너무 많이 반복됨 (발산 가능성)")
            return None

    end_time = time.time()
    execution_time = end_time - start_time

    return x_new, execution_time, step

print("=== Newton Method 성능 측정기 ===") 
fixed_tol = 1e-8 

while True:
    try:
        input_a = float(input("\n초기 추정값(initial guess) a: "))
        result = newton(input_a, fixed_tol)

        if result is not None:
            root, duration, steps = result
            print("-" * 40)
            print(f"✅ 최종 근사값: {root:.10f}")
            print(f"📊 총 반복 횟수: {steps} step")
            print(f"⏱️ 소요 시간: {duration:.8f} 초") 
            print("-" * 40)

            cont = input("계속 하시겠습니까? (y/n): ")
            if cont.lower() not in ['y', 'ㅛ']:
                print("수고하셨습니다")
                break
    except ValueError:
        print("⚠️ 숫자로 입력해 주세요!")