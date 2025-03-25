package Ex0;

import java.util.ArrayList;
import java.util.List;

public class OptimalKinematics {
    public static final double m = 1, k = 1;

    /**
     * Computes the optimal acceleration sequence to move an object from position initPosition to targetPosition
     * while reaching velocity 0 in the shortest possible time.
     *
     * @param initPosition    Initial position of the object.
     * @param initVelocity    Initial velocity of the object.
     * @param targetPosition  Target position where the object must stop.
     * @param maxAcceleration Maximum acceleration magnitude in m/s^2 (must be positive).
     * @param dt              Time step for each iteration (must be positive).
     * @return A list of acceleration values to apply at each time step.
     * @throws IllegalArgumentException If maxAcceleration or dt is non-positive.
     */
    public static List<Double> computeAcceleration
    (double initPosition, double initVelocity, double targetPosition, double maxAcceleration, double dt) {
        if (maxAcceleration <= 0 || dt <= 0) {
            throw new IllegalArgumentException("Acceleration limit maxAcceleration and time step dt must be positive.");
        }

        List<Double> accelerations = new ArrayList<>();
        double distance = targetPosition - initPosition;
        int sign = (distance > 0) ? 1 : -1;

        // Calculate stopping distance considering direction
        double dStop = Math.pow(initVelocity, 2) / (2 * maxAcceleration);
        // Consider if we're moving toward or away from the target
        if (sign * initVelocity > 0) {
            // Moving toward target, stopping distance helps us
            dStop = dStop * sign;
        }
        else if (initVelocity != 0) {
            // Moving away from target, stopping distance hurts us
            dStop = -dStop * sign;
        }

        // Need to check if we can reach the target with just deceleration
        // or if we need to accelerate first
        if (Math.abs(distance) <= Math.abs(dStop) && sign * initVelocity > 0) {
            // If already within stopping distance and moving toward target, decelerate
            double currentPosition = initPosition;
            double currentVelocity = initVelocity;

            while (Math.abs(currentVelocity) > Bereshit_101.EPSILON * dt * maxAcceleration) {
                double deceleration = -sign * Math.min(maxAcceleration, Math.abs(currentVelocity / dt));
//                double desiredDeceleration = -k * currentVelocity - m * (targetPosition - currentPosition);
//                desiredDeceleration = Math.clamp(desiredDeceleration, -maxAcceleration, maxAcceleration);
//                accelerations.add(desiredDeceleration);
                accelerations.add(deceleration);
                currentVelocity += deceleration * dt;
                currentPosition += currentVelocity * dt;

                // If we've passed or reached the target, stop
                if (sign * (targetPosition - currentPosition) <= 0) {
                    break;
                }
            }
        }
        else {
            // Need full acceleration-deceleration profile
            double currentPosition = initPosition;
            double currentVelocity = initVelocity;

            // Compute proper peak velocity (no need to divide by 2 twice)
            double vMax = Math.sqrt(maxAcceleration * Math.abs(distance));

            // Acceleration phase
            boolean accelerationPhase = true;
            while (accelerationPhase) {
                // Calculate stopping distance from current state
                double currentStoppingDistance = Math.pow(currentVelocity, 2) / (2 * maxAcceleration) * sign;
                double remainingDistance = targetPosition - currentPosition;

                if (Math.abs(remainingDistance) <= Math.abs(currentStoppingDistance) && sign * currentVelocity > 0) {
                    // Time to switch to deceleration
                    accelerationPhase = false;
                }
                else if (Math.abs(currentVelocity) >= vMax) {
                    // Reached max velocity
                    accelerationPhase = false;
                }
                else {
                    // Continue accelerating
//                    double desiredAcceleration = -k * currentVelocity - m * (targetPosition - currentPosition);
//                    desiredAcceleration = Math.clamp(desiredAcceleration, -maxAcceleration, maxAcceleration);
                    accelerations.add(sign * maxAcceleration);
                    currentVelocity += sign * maxAcceleration * dt;
                    currentPosition += currentVelocity * dt;
                }
            }

            // Deceleration phase
            while (Math.abs(currentVelocity) > Bereshit_101.EPSILON * dt * maxAcceleration) {
                double deceleration = -sign * Math.min(maxAcceleration, Math.abs(currentVelocity / dt));
//                double deceleration = -sign * Math.min(maxAcceleration, Math.abs(currentVelocity / dt));
                accelerations.add(deceleration);
                currentVelocity += deceleration * dt;
                currentPosition += currentVelocity * dt;

                // If we've passed or reached the target, stop
                if (sign * (targetPosition - currentPosition) <= 0) {
                    break;
                }
            }
        }

        return accelerations;
    }

    public static List<Double> computeAccelerations
            (double initPosition, double initVelocity, double targetPosition, double maxAcceleration, double dt) {
        if (maxAcceleration <= 0 || dt <= 0) {
            throw new IllegalArgumentException("Acceleration limit maxAcceleration and time step dt must be positive.");
        }

        List<Double> accelerations = new ArrayList<>();
        double distance = targetPosition - initPosition;
        int sign = (distance > 0) ? 1 : -1;

        // Calculate stopping distance considering direction
        double dStop = Math.pow(initVelocity, 2) / (2 * maxAcceleration);
        // Consider if we're moving toward or away from the target
        if (sign * initVelocity > 0) {
            // Moving toward target, stopping distance helps us
            dStop = dStop * sign;
        }
        else if (initVelocity != 0) {
            // Moving away from target, stopping distance hurts us
            dStop = -dStop * sign;
        }

        double currentPosition = initPosition;
        double currentVelocity = initVelocity;

        // Need to check if we can reach the target with just deceleration
        // or if we need to accelerate first
        if (Math.abs(distance) <= Math.abs(dStop) && sign * initVelocity > 0) {
            // If already within stopping distance and moving toward target, decelerate


            while (Math.abs(currentVelocity) > Bereshit_101.EPSILON * dt * maxAcceleration) {
                double deceleration = -sign * Math.min(maxAcceleration, Math.abs(currentVelocity / dt));
//                double desiredDeceleration = -k * currentVelocity - m * (targetPosition - currentPosition);
//                desiredDeceleration = Math.clamp(desiredDeceleration, -maxAcceleration, maxAcceleration);
//                accelerations.add(desiredDeceleration);
                accelerations.add(deceleration);
                currentVelocity += deceleration * dt;
                currentPosition += currentVelocity * dt;

                // If we've passed or reached the target, stop
                if (sign * (targetPosition - currentPosition) <= 0) {
                    break;
                }
            }
        }
        else {

            // Compute proper peak velocity (no need to divide by 2 twice)
            double vMax = Math.sqrt(maxAcceleration * Math.abs(distance));

            // Acceleration phase
            boolean accelerationPhase = true;
            while (accelerationPhase) {
                // Calculate stopping distance from current state
                double currentStoppingDistance = Math.pow(currentVelocity, 2) / (2 * maxAcceleration) * sign;
                double remainingDistance = targetPosition - currentPosition;

                if (Math.abs(remainingDistance) <= Math.abs(currentStoppingDistance) && sign * currentVelocity > 0) {
                    // Time to switch to deceleration
                    accelerationPhase = false;
                }
                else if (Math.abs(currentVelocity) >= vMax) {
                    // Reached max velocity
                    accelerationPhase = false;
                }
                else {
                    // Continue accelerating
//                    double desiredAcceleration = -k * currentVelocity - m * (targetPosition - currentPosition);
//                    desiredAcceleration = Math.clamp(desiredAcceleration, -maxAcceleration, maxAcceleration);
                    accelerations.add(sign * maxAcceleration);
                    currentVelocity += sign * maxAcceleration * dt;
                    currentPosition += currentVelocity * dt;
                }
            }

            // Deceleration phase
            while (Math.abs(currentVelocity) > Bereshit_101.EPSILON * dt * maxAcceleration) {
                double deceleration = -sign * Math.min(maxAcceleration, Math.abs(currentVelocity / dt));
//                double deceleration = -sign * Math.min(maxAcceleration, Math.abs(currentVelocity / dt));
                accelerations.add(deceleration);
                currentVelocity += deceleration * dt;
                currentPosition += currentVelocity * dt;

                // If we've passed or reached the target, stop
                if (sign * (targetPosition - currentPosition) <= 0) {
                    break;
                }
            }
        }

        return accelerations;
    }


    /**
     * Simulates the motion of an object and returns the final position and velocity.
     *
     * @param initPosition  Initial position of the object.
     * @param initVelocity  Initial velocity of the object.
     * @param accelerations List of acceleration values to apply at each time step.
     * @param dt            Time step for each iteration.
     * @return An array where [0] is final position and [1] is final velocity.
     */
    public static double[] simulateMotion(double initPosition, double initVelocity,
                                          List<Double> accelerations, double dt) {
        double position = initPosition;
        double velocity = initVelocity;

        for (Double acceleration : accelerations) {
            velocity += acceleration * dt;
            position += velocity * dt;
        }

        return new double[]{position, velocity};
    }

    /**
     * Runs unit tests to verify the computeAcceleration function works correctly.
     * Each test checks if the final position is close to the target and final velocity is close to zero.
     */
    public static void runUnitTests() {
        final double POSITION_TOLERANCE = 0.5;  // Position tolerance in meters
        final double VELOCITY_TOLERANCE = 0.1;  // Velocity tolerance in m/s

        System.out.println("Running unit tests for OptimalKinematics...");

        // Test 1: From 58.4 to -63
        {
            double initPosition = 58.4;
            double initVelocity = 0;
            double targetPosition = -63;
            double maxAcceleration = 4;
            double dt = 0.1;

            List<Double> accelerations = computeAcceleration(
                    initPosition, initVelocity, targetPosition, maxAcceleration, dt);

            double[] finalState = simulateMotion(initPosition, initVelocity, accelerations, dt);
            double finalPosition = finalState[0];
            double finalVelocity = finalState[1];

            boolean positionOK = Math.abs(finalPosition - targetPosition) <= POSITION_TOLERANCE;
            boolean velocityOK = Math.abs(finalVelocity) <= VELOCITY_TOLERANCE;

            System.out.println("\nTest 1: From 58.4 to -63");
            System.out.println("Acceleration steps: " + accelerations.size());
            System.out.println("Final position: " + finalPosition + " (target: " + targetPosition + ")");
            System.out.println("Final velocity: " + finalVelocity);
            System.out.println("Position test: " + (positionOK ? "PASSED" : "FAILED"));
            System.out.println("Velocity test: " + (velocityOK ? "PASSED" : "FAILED"));
        }

        // Test 2: From -63 to 0
        {
            double initPosition = -63;
            double initVelocity = 0;
            double targetPosition = 0;
            double maxAcceleration = 4;
            double dt = 0.1;

            List<Double> accelerations = computeAcceleration(
                    initPosition, initVelocity, targetPosition, maxAcceleration, dt);

            double[] finalState = simulateMotion(initPosition, initVelocity, accelerations, dt);
            double finalPosition = finalState[0];
            double finalVelocity = finalState[1];

            boolean positionOK = Math.abs(finalPosition - targetPosition) <= POSITION_TOLERANCE;
            boolean velocityOK = Math.abs(finalVelocity) <= VELOCITY_TOLERANCE;

            System.out.println("\nTest 2: From -63 to 0");
            System.out.println("Acceleration steps: " + accelerations.size());
            System.out.println("Final position: " + finalPosition + " (target: " + targetPosition + ")");
            System.out.println("Final velocity: " + finalVelocity);
            System.out.println("Position test: " + (positionOK ? "PASSED" : "FAILED"));
            System.out.println("Velocity test: " + (velocityOK ? "PASSED" : "FAILED"));
        }
    }

    public static void main(String[] args) {
        // Run the unit tests with the specific parameters requested
        runUnitTests();
    }
}