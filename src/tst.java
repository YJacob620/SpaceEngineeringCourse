import java.util.Arrays;

/**
 * Implements time-optimal control for a double integrator system with bounded acceleration.
 * This is a "bang-bang" controller that will move an object from current position and velocity
 * to a target position with zero velocity in minimal time.
 */
public class tst {

    /**
     * Computes the time-optimal control for a 1D movement with bounded acceleration.
     *
     * @param currentPosition Current position (x)
     * @param targetPosition  Target position (y)
     * @param currentVelocity Current velocity (v)
     * @param maxAcceleration Maximum acceleration magnitude (c)
     * @return ControlSolution containing the optimal control strategy
     */
    public static ControlSolution computeOptimalControl(
            double currentPosition,
            double targetPosition,
            double currentVelocity,
            double maxAcceleration) {

        if (maxAcceleration <= 0) {
            throw new IllegalArgumentException("Maximum acceleration must be positive");
        }

        // Calculate relative position
        double deltaX = currentPosition - targetPosition;
        double v = currentVelocity;
        double c = maxAcceleration;
        double vSquaredOver2c = (v * v) / (2 * c);

        // Determine the region and compute control parameters
        double switchTime;
        double totalTime;
        double initialControl;

        if (deltaX > vSquaredOver2c) {
            // Region I: far past target or moving away from target
            switchTime = (-v - Math.sqrt(v * v + 2 * c * deltaX)) / c;
            totalTime = (v + Math.sqrt(v * v + 2 * c * deltaX)) / c;
            initialControl = c;  // Start with positive acceleration
        }
        else if (deltaX > 0) {
            // Region II: slightly past target, moving toward it
            switchTime = (v - Math.sqrt(v * v - 2 * c * deltaX)) / c;
            totalTime = (-v - Math.sqrt(v * v - 2 * c * deltaX)) / c;
            initialControl = -c;  // Start with negative acceleration
        }
        else if (deltaX >= -vSquaredOver2c) {
            // Region III: slightly before target, moving toward it
            switchTime = (-v + Math.sqrt(v * v - 2 * c * deltaX)) / c;
            totalTime = (v + Math.sqrt(v * v - 2 * c * deltaX)) / c;
            initialControl = c;  // Start with positive acceleration
        }
        else {
            // Region IV: far before target or moving away from target
            switchTime = (v + Math.sqrt(v * v + 2 * c * deltaX)) / c;
            totalTime = (-v + Math.sqrt(v * v + 2 * c * deltaX)) / c;
            initialControl = -c;  // Start with negative acceleration
        }

        // Ensure switch time is positive (can happen due to numerical issues)
        switchTime = Math.max(0, switchTime);
        totalTime = Math.abs(totalTime); // Ensure total time is positive

        return new ControlSolution(switchTime, totalTime, initialControl);
    }

    /**
     * Simulates the motion of the object using the optimal control.
     * Useful for verifying the algorithm works correctly.
     *
     * @param initialPosition Initial position
     * @param initialVelocity Initial velocity
     * @param solution        Control solution
     * @param maxAcceleration Maximum acceleration magnitude
     * @param dt              Time step for simulation
     * @return Array of positions and velocities at each time step [time, position, velocity]
     */
    public static double[][] simulateMotion(
            double initialPosition,
            double initialVelocity,
            ControlSolution solution,
            double maxAcceleration,
            double dt) {

        int steps = (int) Math.ceil(solution.getTotalTime() / dt) + 1;
        double[][] result = new double[steps][3];  // [time, position, velocity]

        double position = initialPosition;
        double velocity = initialVelocity;
        double time = 0;

        for (int i = 0; i < steps; i++) {
            double control = solution.getControlAtTime(time, maxAcceleration);

            result[i][0] = time;
            result[i][1] = position;
            result[i][2] = velocity;

            // Update states for next step using dynamics equations
            position += velocity * dt + 0.5 * control * dt * dt;
            velocity += control * dt;
            time += dt;
        }

        return result;
    }

    /**
     * Example usage of the algorithm
     */
    public static void main(String[] args) {
        // Example: Object at position 10, moving at 2 m/s, needs to reach position 0
        double currentPosition = 10.0;
        double targetPosition = 0.0;
        double currentVelocity = 2.0;
        double maxAcceleration = 1.0;

        ControlSolution solution = computeOptimalControl(
                currentPosition, targetPosition, currentVelocity, maxAcceleration);

        System.out.println("Optimal Control Solution:");
        System.out.println(solution);

        System.out.println("\nSimulation Results:");
        double[][] simulation = simulateMotion(
                currentPosition, currentVelocity, solution, maxAcceleration, 0.1);

        System.out.println("Time\tPosition\tVelocity");
        // Print some sample points from simulation
        for (int i = 0; i < simulation.length; i += Math.max(1, simulation.length / 10)) {
            System.out.printf("%.1f\t%.4f\t%.4f%n",
                    simulation[i][0], simulation[i][1], simulation[i][2]);
        }

        // Print final state
        double[] finalState = simulation[simulation.length - 1];
        System.out.printf("\nFinal state: Position = %.4f, Velocity = %.4f at time = %.4f\n",
                finalState[1], finalState[2], finalState[0]);
    }
}

/**
 * Contains the solution parameters for the time-optimal control problem.
 */
class ControlSolution {
    private final double switchTime;  // Time at which to switch control
    private final double totalTime;   // Total time to reach target
    private final double initialControl;  // Initial control value (+c or -c)

    public ControlSolution(double switchTime, double totalTime, double initialControl) {
        this.switchTime = switchTime;
        this.totalTime = totalTime;
        this.initialControl = initialControl;
    }

    public double getSwitchTime() {
        return switchTime;
    }

    public double getTotalTime() {
        return totalTime;
    }

    public double getInitialControl() {
        return initialControl;
    }

    /**
     * Gets the control value (acceleration) at a specific time
     *
     * @param time            Current time
     * @param maxAcceleration Maximum acceleration magnitude (unused, kept for clarity)
     * @return Control value (+c, -c, or 0 if time exceeds total time)
     */
    public double getControlAtTime(double time, double maxAcceleration) {
        if (time < 0) {
            return 0;
        }
        else if (time < switchTime) {
            return initialControl;
        }
        else if (time < totalTime) {
            return -initialControl;  // Switch to opposite control
        }
        else {
            return 0;  // Target reached, no more control needed
        }
    }

    @Override
    public String toString() {
        return String.format("Switch from %.2f to %.2f at %.4f seconds, total time: %.4f seconds",
                initialControl, -initialControl, switchTime, totalTime);
    }
}