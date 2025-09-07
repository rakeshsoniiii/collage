import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

#  Define fuzzy variables
temperature = ctrl.Antecedent(np.arange(0, 41, 1), 'temperature')  # Celsius
humidity = ctrl.Antecedent(np.arange(0, 101, 1), 'humidity')        # %
fan_speed = ctrl.Consequent(np.arange(0, 101, 1), 'fan_speed')      # %

#  Membership functions

# Temperature
temperature['low'] = fuzz.trimf(temperature.universe, [0, 0, 20])
temperature['medium'] = fuzz.trimf(temperature.universe, [15, 25, 35])
temperature['high'] = fuzz.trimf(temperature.universe, [30, 40, 40])

# Humidity
humidity['low'] = fuzz.trimf(humidity.universe, [0, 0, 40])
humidity['medium'] = fuzz.trimf(humidity.universe, [30, 50, 70])
humidity['high'] = fuzz.trimf(humidity.universe, [60, 100, 100])

# Fan Speed
fan_speed['slow'] = fuzz.trimf(fan_speed.universe, [0, 0, 50])
fan_speed['moderate'] = fuzz.trimf(fan_speed.universe, [30, 50, 70])
fan_speed['fast'] = fuzz.trimf(fan_speed.universe, [60, 100, 100])

#  Define rules
rules = [
    ctrl.Rule(temperature['high'] & humidity['high'], fan_speed['fast']),
    ctrl.Rule(temperature['high'] & humidity['medium'], fan_speed['fast']),
    ctrl.Rule(temperature['high'] & humidity['low'], fan_speed['moderate']),

    ctrl.Rule(temperature['medium'] & humidity['high'], fan_speed['fast']),
    ctrl.Rule(temperature['medium'] & humidity['medium'], fan_speed['moderate']),
    ctrl.Rule(temperature['medium'] & humidity['low'], fan_speed['moderate']),

    ctrl.Rule(temperature['low'] & humidity['high'], fan_speed['moderate']),
    ctrl.Rule(temperature['low'] & humidity['medium'], fan_speed['slow']),
    ctrl.Rule(temperature['low'] & humidity['low'], fan_speed['slow']),
]

#  Control system
fan_ctrl = ctrl.ControlSystem(rules)
fan_sim = ctrl.ControlSystemSimulation(fan_ctrl)

#  Input values
fan_sim.input['temperature'] = 20
fan_sim.input['humidity'] = 35

#  Compute output
try:
    fan_sim.compute()
    speed = fan_sim.output['fan_speed']
    print(f"Temperature: 20°C, Humidity: 35% → Fan Speed: {speed:.2f}%")
except Exception as e:
    print("Error computing fan speed:", e)

#  Visualizations
temperature.view()
humidity.view()
fan_speed.view()
fan_speed.view(sim=fan_sim)

plt.show()
