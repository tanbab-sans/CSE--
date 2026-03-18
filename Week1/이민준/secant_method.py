import time 

def f(x):
    return x**5 - 9*x**4 - x**3 + 17*x**2 - 8*x - 8

def secant(a, b, tolerance):

    if f(a) == 0: return a, 0.0, 0
    if f(b) == 0: return b, 0.0, 0

    step = 1
    start_time = time.time() 

    while True:

        if f(a) - f(b) == 0:
            print("❌ 분모가 0이 되어 계산할 수 없습니다.")
            return None

        x_new = a - f(a) * ((a - b) / (f(a) - f(b)))

        print(f"Step {step}: x_{step-1}=[{a:.8f}], x_{step}=[{b:.8f}] -> New=[{x_new:.8f}]")

        if abs(x_new - b) < tolerance:
            break

        a = b
        b = x_new

        step += 1

        if step > 100: 
            print("⚠️ 100번 넘게 돌았는데 안 모여요! (발산 가능성)")
            return None

    end_time = time.time()
    execution_time = end_time - start_time

    return x_new, execution_time, step

print("=== Secant Method 성능 측정기 ===")
fixed_tol = 1e-8 

while True:
    try:

        input_a = float(input("\n첫 번째 추정값 (x1) a: "))
        input_b = float(input("두 번째 추정값 (x2) b: "))

        result = secant(input_a, input_b, fixed_tol)

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