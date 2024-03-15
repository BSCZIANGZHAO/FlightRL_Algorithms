class PIDController:
    def __init__(self, P, I, D, target):
        self.P = P
        self.I = I
        self.D = D
        self.target = target
        self.integral = 0
        self.prev_error = None

    def update(self, current_value):
        """Update the PID controller with the current value and return the control signal"""
        # Error
        error = self.target - current_value

        # Integral term
        self.integral += error

        # Derivative term
        derivative = 0 if self.prev_error is None else error - self.prev_error

        # Control signal
        control_signal = self.P * error + self.I * self.integral + self.D * derivative

        # Update previous error
        self.prev_error = error

        return control_signal

    def reset(self):
        """
        Reset the integral and previous error
        """
        self.integral = 0
        self.prev_error = None
