import time  

def f(x):
    return x**5 - 9*x**4 - x**3 + 17*x**2 - 8*x - 8

def bisection(a, b, tolerance):
    if f(a) == 0: return a , 0.0 , 0
    if f(b) == 0: return b , 0.0 , 0
    if f(a) * f(b) > 0:
        print("\n❌ [오류] f(a)와 f(b)의 부호가 같습니다.")
        return None

    step = 1

    start_time = time.time() 

    while (b - a) / 2 > tolerance:
        m = (a + b) / 2
        f_m = f(m)
        if f_m == 0: break
        elif f(a) * f_m < 0: b = m
        else: a = m
        print(f"Step {step}: 구간=[{a:.8f}, {b:.8f}], 중간값={m:.8f}")
        step += 1

    end_time = time.time()

    execution_time = end_time - start_time
    return (a + b) / 2, execution_time, step

print("=== Bisection Method 성능 측정기 ===")
fixed_tol = 1e-8 

while True:
    try:
        input_a = float(input("\n시작점 a: "))
        input_b = float(input("끝점 b: "))

        result = bisection(input_a, input_b, fixed_tol)

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