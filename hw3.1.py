import numpy as np

#напишіть функцію гіпотези лінійної регресії у векторному вигляді;
def hypothesis(X, w):
    return np.dot(X, w)

#створіть функцію для обчислення функції втрат у векторному вигляді;
def compute_cost(X, y, w):
    m = len(y)
    predictions = hypothesis(X, w)
    cost = (1/(2*m)) * np.sum(np.square(predictions - y))
    return cost

#реалізуйте один крок градієнтного спуску;
def gradient_descent_step(X, y, w, learning_rate):
    m = len(y)
    predictions = hypothesis(X, w)
    gradient = (1/m) * np.dot(X.T, (predictions - y))
    w -= learning_rate * gradient
    return w

#знайдіть найкращі параметри $\vec{w}$ для датасету прогнозуючу ціну на будинок залежно від площі, кількості ванних кімнат та кількості спалень;
# ініціалізація ваг та інших параметрів
w = np.zeros(X.shape[1])  # ініціалізуємо ваги нулями
learning_rate = 0.01
iterations = 1000

for i in range(iterations):
    w = gradient_descent_step(X, y, w, learning_rate)

final_cost = compute_cost(X, y, w)

#знайдіть ці ж параметри за допомогою аналітичного рішення;
def analytical_solution(X, y):
    return np.linalg.inv(X.T @ X) @ X.T @ y

w_analytical = analytical_solution(X, y)

#порівняйте отримані результати.
# після завершення градієнтного спуску
w_gradient_descent = w 

# за допомогою аналітичного рішення
w_analytical_solution = analytical_solution(X, y)  

print("Градієнтний спуск:", w_gradient_descent)
print("Аналітичне рішення:", w_analytical_solution)

# порівняйте значення втрат
cost_gradient_descent = compute_cost(X, y, w_gradient_descent)
cost_analytical_solution = compute_cost(X, y, w_analytical_solution)
print("Функція втрат для градієнтного спуску:", cost_gradient_descent)
print("Функція втрат для аналітичного рішення:", cost_analytical_solution)
