package Ex0;


public class Bereshit_101 {
    public static final double DT = 1; // seconds

    /* Spacecraft parameters */
    public static final double MASS_WHEN_EMPTY = 165; // kg
    public static final double MAIN_ENGINE_MAX_FORCE = 430; // Newton
    public static final double MAIN_ENGINE_MAX_FUEL_BURN = 0.15; // liters per sec

    static final int BALANCE_ENGINES_PER_SIDE = 4;
    public static final double BALANCE_ENGINES_MAX_FORCE = 25 * BALANCE_ENGINES_PER_SIDE; // Newton
    public static final double BALANCE_ENGINE_MAX_FUEL_BURN = 0.009 * BALANCE_ENGINES_PER_SIDE; // liters per sec
    public static final double BALANCE_ENGINES_MAX_ANGULAR_ACC = (1 * BALANCE_ENGINES_PER_SIDE); // degrees per sec^2

    /* Starting point parameters (default case - 33:01 in the video) */
    public static double INITIAL_ALTITUDE = 13748; // meters
    public static double INITIAL_FUEL_MASS = 121.06; // liters
    public static double INITIAL_VERTICAL_VELOCITY = 24.8; // m/s
    public static double INITIAL_HORIZONTAL_VELOCITY = 932.2; // m/s
    public static double INITIAL_ANGLE = 58.4; // 2-dimensional angle (degrees)
    public static double INITIAL_DISTANCE = 181055; // meters
    public static final double EPSILON = 0.0001; //

    public static class Spacecraft {
        public double verticalVelocity = INITIAL_VERTICAL_VELOCITY;
        public double horizontalVelocity = INITIAL_HORIZONTAL_VELOCITY;
        public double distance = INITIAL_DISTANCE;
        public double angle = INITIAL_ANGLE; // 0 is parallel to the ground (as in landing)
        public double spinVelocity = 0; // degrees per second
        public double altitude = INITIAL_ALTITUDE;
        public double fuelMass = INITIAL_FUEL_MASS;
        public double totalMass = MASS_WHEN_EMPTY + fuelMass;

        @Override
        public String toString() {
            return String.format(
                    "altitude = %.2f m, distance = %.2f m, v-velocity = %.2f m/s, h-velocity = %.2f m/s, " +
                            "angle = %.2f°, spin = %.2f°/s, fuel: %.2f liters",
                    altitude, distance,
                    verticalVelocity, horizontalVelocity,
                    angle, spinVelocity,
                    fuelMass
            );
        }

        /**
         * Applies the gravitational force of the Moon to the spacecraft,
         * modifying its vertical velocity.
         */
        void applyMoonGravity() {
            verticalVelocity += Moon.getVerticalAccelerationOnObject(horizontalVelocity, altitude) * DT;
        }

        /**
         * Applies forces from the spacecraft's main and balance engines.
         *
         * @param mainPower  Force from the main engine (Newton).
         * @param frontPower Force from the front balance engine (Newton).
         * @param backPower  Force from the back balance engine (Newton).
         */
        public void applyEnginesForces(double mainPower, double frontPower, double backPower) {
            applyEnginesNormalizedForces(
                    mainPower / MAIN_ENGINE_MAX_FORCE,
                    frontPower / BALANCE_ENGINES_MAX_FORCE,
                    backPower / BALANCE_ENGINES_MAX_FORCE);
        }

        public void applyEnginesNormalizedForces(double mainNormalizedPower, double frontNormalizedPower, double backNormalizedPower) {
            // Normalize input powers
            mainNormalizedPower = Math.clamp(mainNormalizedPower, 0, 1);
            frontNormalizedPower = Math.clamp(frontNormalizedPower, 0, 1);
            backNormalizedPower = Math.clamp(backNormalizedPower, 0, 1);

            // Skip calculation if all engines are off
            if (mainNormalizedPower <= 0 && frontNormalizedPower <= 0 && backNormalizedPower <= 0) return;

            // Calculate forces
            double mainForce = mainNormalizedPower * MAIN_ENGINE_MAX_FORCE;
            double frontForce = frontNormalizedPower * BALANCE_ENGINES_MAX_FORCE;
            double backForce = backNormalizedPower * BALANCE_ENGINES_MAX_FORCE;

            // Calculate fuel consumption
            double mainFuelBurned = mainNormalizedPower * MAIN_ENGINE_MAX_FUEL_BURN * DT;
            double frontFuelBurned = frontNormalizedPower * BALANCE_ENGINE_MAX_FUEL_BURN * DT;
            double backFuelBurned = backNormalizedPower * BALANCE_ENGINE_MAX_FUEL_BURN * DT;
            double totalFuelBurned = mainFuelBurned + frontFuelBurned + backFuelBurned;

            // Check if we have enough fuel
            if (fuelMass < totalFuelBurned) {
                // Adjust forces proportionally based on available fuel
                double fuelRatio = fuelMass / totalFuelBurned;
                mainForce *= fuelRatio;
                frontForce *= fuelRatio;
                backForce *= fuelRatio;
                totalFuelBurned = fuelMass;
            }

            // Update fuel mass
            fuelMass -= totalFuelBurned;

            // Calculate acceleration components
            double angleRadians = Math.toRadians(angle + 360);
            // a = f/m
            double mainAccelerationY = -(mainForce * Math.cos(angleRadians)) / totalMass;
            double mainAccelerationX = (mainForce * Math.sin(angleRadians)) / totalMass;
            double frontAccelerationY = -(frontForce * Math.cos(angleRadians)) / totalMass;
            double frontAccelerationX = (frontForce * Math.sin(angleRadians)) / totalMass;
            double backAccelerationY = -(backForce * Math.cos(angleRadians)) / totalMass;
            double backAccelerationX = (backForce * Math.sin(angleRadians)) / totalMass;
            // Combine all accelerations
            double totalAccelerationY = mainAccelerationY + frontAccelerationY + backAccelerationY;
            double totalAccelerationX = mainAccelerationX + frontAccelerationX + backAccelerationX;

            // Update velocities
            verticalVelocity += totalAccelerationY * DT;
            horizontalVelocity += totalAccelerationX * DT;

            // Calculate angular effects from balance engines
            // Front engines cause counter-clockwise rotation (negative angle change)
            double frontAngularAcceleration = -frontNormalizedPower * BALANCE_ENGINES_MAX_ANGULAR_ACC;

            // Back engines cause clockwise rotation (positive angle change)
            double backAngularAcceleration = backNormalizedPower * BALANCE_ENGINES_MAX_ANGULAR_ACC;

            // Net angular acceleration
            double netAngularAcceleration = frontAngularAcceleration + backAngularAcceleration;

            // Update spin velocity
            spinVelocity += netAngularAcceleration * DT;
        }

        /**
         * Updates the spacecraft's state based on its velocity, position, and fuel consumption.
         */
        void update() {
            applyMoonGravity();
            altitude -= verticalVelocity * DT;
            distance -= horizontalVelocity * DT;
            angle += spinVelocity * DT;
            totalMass = MASS_WHEN_EMPTY + fuelMass;
        }

        /**
         * @return The force (in Newtons) required for vertical-velocity to be 0 at altitude 0
         */
        double getForceForLanding() {
            double ans = 0;
            double distance = altitude;
            double velocity = verticalVelocity;
            double maxPossibleForce = getMaxEnginesForce(true) - Moon.GRAVITY_CONST * totalMass;

            if (Math.abs(distance) < EPSILON) {
                double mainForce = getForceForVerticalDeceleration(velocity);
                ans = mainForce;
                return ans;
            }

            // Calculate stopping distance
            // f = ma  ->  a = f/m  ->  1/a = m/f
            // d_stop = 0.5 * m/f * v^2
            double currStopDistance = 0.5 * (totalMass / maxPossibleForce) * velocity * velocity;
            double nextVelocity = velocity + Moon.GRAVITY_CONST * DT;
            double nextDistance = Math.abs(distance - nextVelocity * DT);
            double nextStopDistance = 0.5 * (totalMass / maxPossibleForce) * nextVelocity * nextVelocity;
            if (Math.abs(distance) < currStopDistance) {
                System.out.println("GONNA CRASH!");
            }
//            if ((nextDistance < nextStopDistance) ||
//                    (nextDistance < currStopDistance)) { // then decelerate
            if (nextDistance < nextStopDistance) { // then decelerate
                // assume we cannot stop AFTER the target, only BEFORE
                double reqForce = (velocity * velocity * totalMass) / (2 * Math.abs(distance));
                reqForce += Moon.getForceOnObject(totalMass);
                ans = reqForce;
            }
            return ans;
        }

        boolean lastDtAccelerated = true;
        double currentTargetAngle = Double.MAX_VALUE;
        public boolean finishedReachingAngle = false;

        double[] balanceEnginesForcesForTarget(double target) {
            if (Math.abs(currentTargetAngle - target) > EPSILON) {
                currentTargetAngle = target;
                lastDtAccelerated = true;
            }

            double[] ans = new double[2]; // [frontEngineForce, backEngineForce]
            double velocity = spinVelocity;
            double distance = (target - angle);
            boolean movingAwayFromTarget = velocity != 0 && distance * velocity < 0;
            double maxForce = BALANCE_ENGINES_MAX_FORCE;

            if (Math.abs(distance) < EPSILON) {
                // assume we can reach to a stop in 1 DT at this point
                double balanceForce = Math.abs(velocity) / (BALANCE_ENGINES_MAX_ANGULAR_ACC * DT) * maxForce;
                if (velocity > 0) {
                    ans[0] = balanceForce;
                }
                else {
                    ans[1] = balanceForce;
                }
                finishedReachingAngle = true;
                return ans;
            }

            if (movingAwayFromTarget) { // then full force brake spin
                if (velocity > 0) {
                    ans[0] = maxForce;
                }
                else {
                    ans[1] = maxForce;
                }
                return ans;
            }
            /* From here we're moving towards target */


            double reqForce = maxForce;
            double nextVelocityIfAccelerating;
            if (velocity > 0) {
                nextVelocityIfAccelerating = velocity + BALANCE_ENGINES_MAX_ANGULAR_ACC * DT;
            }
            else {
                nextVelocityIfAccelerating = velocity - BALANCE_ENGINES_MAX_ANGULAR_ACC * DT;
            }
            boolean shouldDecelerate = false;
            double nextDistanceIfAccelerating = distance - nextVelocityIfAccelerating * DT;
            // Calculate stopping distance
            // d_stop = 0.5 * 1/a * v^2
            double nextStopDistanceIfAccelerating =
                    0.5 * (1 / BALANCE_ENGINES_MAX_ANGULAR_ACC) * Math.pow(nextVelocityIfAccelerating, 2);
            if (nextDistanceIfAccelerating * distance < 0) { // If we'll go over the target without stopping
                shouldDecelerate = true;
            }
            nextDistanceIfAccelerating = Math.abs(nextDistanceIfAccelerating);
            if (!shouldDecelerate) {
                shouldDecelerate = nextDistanceIfAccelerating < nextStopDistanceIfAccelerating;
            }
            if (shouldDecelerate) { // then decelerate
                // a = v^2/(2*d_stop)
                double reqDeceleration = (velocity * velocity) / (2 * Math.abs(distance));
                reqForce = (reqDeceleration / BALANCE_ENGINES_MAX_ANGULAR_ACC) * maxForce;
                if (velocity > 0) {
                    ans[0] = reqForce;
                }
                else {
                    ans[1] = reqForce;
                }
                lastDtAccelerated = false;
            }
            else { // then accelerate
                if (!lastDtAccelerated && Math.abs(velocity) > EPSILON) { // don't accelerate if deceleration has started
                    return ans;
                }
                if (distance < 0) {
                    ans[0] = reqForce;
                }
                else {
                    ans[1] = reqForce;
                }
                lastDtAccelerated = true;
            }
            return ans;
        }

        /**
         * Gets the amount of force (in Newtons) required to make the
         * spacecraft decelerate vertically (when falling) by a given amount
         *
         * @param requiredDeceleration By how much to decelerate (m/s^2)
         * @return Force required to reach wanted deceleration when falling.
         */
        double getForceForVerticalDeceleration(double requiredDeceleration) {
            // f = ma
            double ans = totalMass * (requiredDeceleration + Moon.GRAVITY_CONST);
            return ans;
        }

        /**
         * @param vertical Whether to calculate vertical or horizontal forces
         * @return The maximum amount of force the engines can output in a given direction
         */
        double getMaxEnginesForce(boolean vertical) {
            double maxForce = (BALANCE_ENGINES_MAX_FORCE * 2 + MAIN_ENGINE_MAX_FORCE);
            if (vertical) {
                maxForce *= Math.cos(Math.toRadians(angle + 360)); // only vertical force
            }
            else {
                maxForce *= Math.sin(Math.toRadians(angle + 360)); // only horizontal force
            }
            return maxForce;
        }
    }

    enum States {SET_ANGLE_1, SET_H_VELOCITY, SET_ANGLE_2, SET_V_VELOCITY, NOTHING}

    /**
     * Runs the spacecraft landing simulation with the given parameters.
     *
     * @param angle1       The target angle to reach before starting to adjust horizontal velocity.
     * @param hVelocityReq The maximum horizontal velocity allowed at landing
     *                     (sometimes better when greater than 0, depends on DT).
     * @param angle2       The target angle to reach before starting to adjust vertical velocity for landing.
     */
    public static void runSimulation(double angle1, double hVelocityReq, double angle2) {
        Spacecraft spacecraft = new Spacecraft();
        double totalTime = 0;
        int timeJumps = 0;
        States state = States.SET_ANGLE_1;

        System.out.println("Simulating Bereshit's Landing:");
        System.out.printf("time: %.2f, %s%n", totalTime, spacecraft);

        while (spacecraft.altitude > 0) {
            if (timeJumps % 10 == 0 || spacecraft.altitude < 100) {
                System.out.printf("time: %.2f, %s%n", totalTime, spacecraft);
            }

            double[] balanceEnginesReqForces = new double[2];
            double mainEngineReqForce = 0;
            if (state == States.SET_ANGLE_1) {
                balanceEnginesReqForces = spacecraft.balanceEnginesForcesForTarget(angle1);
            }
            else if (state == States.SET_H_VELOCITY) {
                if (spacecraft.horizontalVelocity > hVelocityReq) {
                    mainEngineReqForce = MAIN_ENGINE_MAX_FORCE;
                    balanceEnginesReqForces = new double[]{BALANCE_ENGINES_MAX_FORCE, BALANCE_ENGINES_MAX_FORCE};
                }
            }
            else if (state == States.SET_ANGLE_2) {
                balanceEnginesReqForces = spacecraft.balanceEnginesForcesForTarget(angle2);
            }
            else if (state == States.SET_V_VELOCITY) {
                mainEngineReqForce = spacecraft.getForceForLanding();
                if (mainEngineReqForce > MAIN_ENGINE_MAX_FORCE) {
                    double balanceOutputs = (mainEngineReqForce - MAIN_ENGINE_MAX_FORCE) / 2;
                    balanceEnginesReqForces[0] = balanceOutputs;
                    balanceEnginesReqForces[1] = balanceOutputs;
                }
            }
            spacecraft.applyEnginesForces(mainEngineReqForce, balanceEnginesReqForces[0], balanceEnginesReqForces[1]);
            spacecraft.update();

            if (state == States.SET_ANGLE_1 && spacecraft.finishedReachingAngle) {
                spacecraft.finishedReachingAngle = false;
                state = States.SET_H_VELOCITY;
            }
            else if (state == States.SET_H_VELOCITY && spacecraft.horizontalVelocity <= hVelocityReq) {
                state = States.SET_ANGLE_2;
            }
            else if (state == States.SET_ANGLE_2 && spacecraft.finishedReachingAngle) {
                spacecraft.finishedReachingAngle = false;
                state = States.SET_V_VELOCITY;
            }

            timeJumps += 1;
            totalTime += DT;
        }
        System.out.printf("time: %.2f, %s%n", totalTime, spacecraft);
    }

    public static void main(String[] args) {
        /* alternative case - 26:52 in the video (uncomment to use) */

//        INITIAL_ALTITUDE = 22629;
//        INITIAL_FUEL_MASS = 197.06;
////        INITIAL_ANGLE = ?;
//        INITIAL_HORIZONTAL_VELOCITY = 1564.9;
//        INITIAL_VERTICAL_VELOCITY = 22.2;

        AltitudeToAngleInterpolator interpolator = new AltitudeToAngleInterpolator();
        double angle1 = interpolator.interpolate(INITIAL_ALTITUDE);

        runSimulation(angle1, 3, 0);
    }
}

/**
 * Simple interpolator for imitating a linear function that gets the spacecraft's altitude
 * and returns a good angle1 for it to use in runSimulation().
 */
class AltitudeToAngleInterpolator {
    private final double[] xValues = {0, 13748, 22629};  // Add more as needed
    private final double[] yValues = {0, -63, -70};      // Corresponding values

    public double interpolate(double x) {
        for (int i = 0; i < xValues.length - 1; i++) {
            if (x >= xValues[i] && x <= xValues[i + 1]) {
                double x1 = xValues[i], x2 = xValues[i + 1];
                double y1 = yValues[i], y2 = yValues[i + 1];
                return y1 + (y2 - y1) * (x - x1) / (x2 - x1); // Linear interpolation formula
            }
        }
        throw new IllegalArgumentException("Value out of interpolation range.");
    }
}