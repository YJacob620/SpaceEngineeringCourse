package Ex0;


public class Bereshit_101 {
    public static final double DT = 0.1; // seconds

    /* Spacecraft parameters */
    public static final double MASS_WHEN_EMPTY = 165; // kg
    public static final double MAIN_ENGINE_MAX_FORCE = 430; // Newton
    public static final double MAIN_ENGINE_MAX_FUEL_BURN = 0.15; // liters per sec

    static final int BALANCE_ENGINES_PER_SIDE = 4;
    public static final double BALANCE_ENGINES_MAX_FORCE = 25 * BALANCE_ENGINES_PER_SIDE; // Newton
    public static final double BALANCE_ENGINE_MAX_FUEL_BURN = 0.009 * BALANCE_ENGINES_PER_SIDE; // liters per sec
    public static final double BALANCE_ENGINES_MAX_ANGULAR_ACC = (1 * BALANCE_ENGINES_PER_SIDE); // degrees per sec^2

    /* Starting point parameters */
    public static final double INITIAL_ALTITUDE = 13748; // meters
    public static final double INITIAL_FUEL_MASS = 121.06; // liters
    public static final double INITIAL_VERTICAL_VELOCITY = 24.8; // m/s
    public static final double INITIAL_HORIZONTAL_VELOCITY = 932.2; // m/s
    public static final double INITIAL_ANGLE = 58.4; // 2-dimensional angle (degrees)
    public static final double INITIAL_DISTANCE = 181055; // meters
    public static final double EPSILON = 0.001; //

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

        public void applyMoonGravity() {
            verticalVelocity += Moon.getVerticalAccelerationOnObject(horizontalVelocity, altitude) * DT;
        }

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

        void update() {
            applyMoonGravity();
            altitude -= verticalVelocity * DT;
            distance -= horizontalVelocity * DT;
            angle += spinVelocity * DT;
            totalMass = MASS_WHEN_EMPTY + fuelMass;
        }

        /**
         * Calculates how much power the front and back balance-engines should use so that the spacecraft will
         * reach a given angle. Note that the returned values are normalized in [0,1].
         *
         * @param targetAngle Required spacecraft angle
         * @return A double array of length 2:
         * <p>
         * [0] - amount of power required from front engine.
         * </p>
         * <p>
         * [1] - amount of power required from back engine.
         * </p>
         */
        double[] engineForcesForAngle(double targetAngle) {
            double[] ans = new double[2]; // [frontEngineForce, backEngineForce]
            double angleDiff = targetAngle - angle;
            double requiredFrontAngularAcc = 0, requiredBackAngularAcc = 0;

            if (Math.abs(angleDiff) < EPSILON) {
                if (spinVelocity > 0) {
                    requiredFrontAngularAcc = spinVelocity;
                }
                else {
                    requiredBackAngularAcc = -spinVelocity;
                }
            }
            else if (angleDiff < 0) {
                if (spinVelocity >= 0) {
                    requiredFrontAngularAcc = Math.abs(spinVelocity) + Math.abs(angleDiff);
                }
                else if (Math.abs(spinVelocity) > Math.abs(angleDiff)) {
                    requiredBackAngularAcc = Math.abs(spinVelocity - angleDiff);
                }

            }
            else if (angleDiff > 0) {
                if (spinVelocity <= 0) {
                    requiredBackAngularAcc = Math.abs(spinVelocity) + Math.abs(angleDiff);
                }
                else if (Math.abs(spinVelocity) > Math.abs(angleDiff)) {
                    requiredFrontAngularAcc = Math.abs(spinVelocity - angleDiff);
                }
            }
            ans[0] = requiredFrontAngularAcc / BALANCE_ENGINES_MAX_ANGULAR_ACC;
            ans[1] = requiredBackAngularAcc / BALANCE_ENGINES_MAX_ANGULAR_ACC;
            return ans;
        }


        double[] engineForcesForTarget(double target, boolean balance) {
            double[] ans = new double[3]; // [mainEngineForce, frontEngineForce, backEngineForce]
            double velocity = balance ? spinVelocity : verticalVelocity;
            double distance = balance ? (target - angle) : altitude;
            boolean movingAwayFromTarget = velocity != 0 && distance * velocity < 0;
            double maxForce = balance ? BALANCE_ENGINES_MAX_FORCE : getMaxEnginesForce(true);

            if (Math.abs(distance) < EPSILON) {
                if (balance) {
                    double balanceForce = Math.abs(velocity) / (BALANCE_ENGINES_MAX_ANGULAR_ACC * DT) * maxForce;
                    if (velocity > 0) {
                        ans[1] = balanceForce;
                    }
                    else {
                        ans[2] = balanceForce;
                    }
                }
                else {
                    double mainForce = getForceForVerticalDeceleration(velocity);
                    ans[1] = mainForce;
                }
                return ans;
            }

            if (balance) {
                if (movingAwayFromTarget) { // then brake spin
                    if (velocity > 0) {
                        ans[1] = maxForce;
                    }
                    else {
                        ans[2] = maxForce;
                    }
                    return ans;
                }
            }

            // Calculate stopping distance
            // f = ma  ->  a = f/m  ->  1/a = m/f
            double stopDistance;
            if (balance) {
                // d_stop = 0.5 * 1/a * v^2
                stopDistance = 0.5 * (1 / BALANCE_ENGINES_MAX_ANGULAR_ACC) * spinVelocity * spinVelocity;
            }
            else {
                // d_stop = 0.5 * m/f * v^2
                // f = enginesMaxPower - moonForceOnSpaceship
                // moonForceOnSpaceship = gravity_const * mass
                stopDistance = 0.5 * (totalMass / (maxForce - Moon.GRAVITY_CONST * totalMass))
                        * verticalVelocity * verticalVelocity;
            }

            double acceleration;
            if (balance) {
                acceleration = BALANCE_ENGINES_MAX_ANGULAR_ACC;
            }
            else {
                acceleration = getCurrentMaxVerticalDeceleration();
            }
            double reqForce = maxForce;
            double nextVelocity = velocity + acceleration * DT;
            if (Math.abs(distance - nextVelocity * DT) < stopDistance) { // then decelerate
                // assume we cannot stop AFTER the target, only BEFORE.
                // check if the next deceleration will cause to stop BEFORE target
//                if (nextVelocity * velocity < 0) { // if we'll skip the stop point
//                    // f = (v^2*m)/(2*d_stop)
                reqForce = (velocity * velocity * totalMass) / (2 * Math.abs(distance));
//                }
                if (balance) {
                    if (velocity > 0) {
                        ans[1] = reqForce;
                    }
                    else {
                        ans[2] = reqForce;
                    }
                }
                else {
                    if (reqForce > maxForce) {
                        ans[0] = maxForce;
                        ans[1] = reqForce - maxForce;
                        ans[2] = ans[1];
                    }
                }
                lastDtAccelerated = false;
            }
            else { // then accelerate (main engine should not accelerate fall)
//                if (Math.abs(distance + velocity * DT) < stopDistance) {
//                    if (Math.abs(distance + nextVelocity * DT) > stopDistance) {
//                        return ans;
//                    }
//                }
                if (!lastDtAccelerated) {
                    return ans;
                    // f = (v^2*m)/(2*d_stop)
//                    double tmp = (velocity * velocity * totalMass) / (2 * Math.abs(distance));
//                    reqForce = Math.min(reqForce, tmp);
                }

                if (balance) {
                    if (distance < 0) {
                        ans[1] = reqForce;
                    }
                    else {
                        ans[2] = reqForce;
                    }
                }
                lastDtAccelerated = true;
            }

            return ans;
        }

        /**
         * @return The force required for vertical-velocity to be 0 at altitude 0
         */
        double mainEngineForceForLanding() {
            double ans = 0;
            double velocity = verticalVelocity;
            double distance = altitude;
            double maxPossibleForce = getMaxEnginesForce(true);
            // moonForceOnSpaceship = gravity_const * mass
            maxPossibleForce -= Moon.GRAVITY_CONST * totalMass;
            double maxPossibleDeceleration = getMaxEnginesForce(true) / totalMass;

            if (Math.abs(distance) < EPSILON) {
                double mainForce = getForceForVerticalDeceleration(velocity);
                ans = mainForce;
                return ans;
            }

            // Calculate stopping distance
            // f = ma  ->  a = f/m  ->  1/a = m/f
            // d_stop = 0.5 * m/f * v^2
            // f = enginesMaxPower - moonForceOnSpaceship
            double stopDistance = 0.5 * (totalMass / maxPossibleForce) * velocity * velocity;
            double nextVelocity = velocity + Moon.GRAVITY_CONST * DT;
            double nextDistance = Math.abs(distance - nextVelocity * DT);
            double nextStopDistance = 0.5 * (totalMass / maxPossibleForce) * nextVelocity * nextVelocity;
            if ((nextDistance < nextStopDistance) ||
                    (nextDistance < stopDistance)) { // then decelerate
                // assume we cannot stop AFTER the target, only BEFORE
                double reqForce = MAIN_ENGINE_MAX_FORCE + 2 * BALANCE_ENGINES_MAX_FORCE;
                double reqForce2 = (velocity * velocity * totalMass) / (2 * Math.abs(distance));
                reqForce2 += Moon.getForceOnObject(totalMass);
//                reqForce *= (nextDistance / nextStopDistance);
//                double reqForceNormalized = nextDistance / stopDistance;
//                ans = reqForce;
                ans = reqForce2;
//                ans = MAIN_ENGINE_MAX_FORCE;
//                ans = reqForceNormalized * maxForce;
            }


            return ans;
        }

        static boolean lastDtAccelerated = true;
        static double currentTargetAngle = Double.MAX_VALUE;

        double[] balanceEnginesForcesForTarget(double target) {
            if (Math.abs(currentTargetAngle - target) > EPSILON) {
                currentTargetAngle = target;
                lastDtAccelerated = true;
            }

            double[] ans = new double[2]; // [mainEngineForce, frontEngineForce, backEngineForce]
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
                return ans;
            }

            if (movingAwayFromTarget) { // then brake spin
                if (velocity > 0) {
                    ans[0] = maxForce;
                }
                else {
                    ans[1] = maxForce;
                }
                return ans;
            }

            // Calculate stopping distance
            // f = ma  ->  a = f/m  ->  1/a = m/f
            // d_stop = 0.5 * m/f * v^2 = 0.5 * 1/a * v^2
            double reqForce = maxForce;
            double nextVelocity = velocity + BALANCE_ENGINES_MAX_ANGULAR_ACC * DT;
            double nextDistance = Math.abs(distance - nextVelocity * DT);
            double stopDistance = 0.5 * (1 / BALANCE_ENGINES_MAX_ANGULAR_ACC) * velocity * velocity;
            if (nextDistance < stopDistance) { // then decelerate
                // deceleration is relative to (nextDistance/stopDistance)
                double decelerationNormalized = nextDistance / stopDistance;
                reqForce = decelerationNormalized * maxForce;
                if (velocity > 0) {
                    ans[0] = reqForce;
                }
                else {
                    ans[1] = reqForce;
                }
                lastDtAccelerated = false;
            }
            else { // then accelerate
                if (!lastDtAccelerated) { // don't accelerate if deceleration has started
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

        /**
         * @return (non - negative)
         */
        public double getCurrentMaxVerticalDeceleration() {
            // a = f/m
            double maxDec = getMaxEnginesForce(true) / totalMass;
            maxDec -= Moon.GRAVITY_CONST;
            return maxDec;
        }

        double[] engineForcesForVerticalVelocityWhenAngleZero(double targetVelocity) {
            double[] ans = new double[2];

            if (verticalVelocity > targetVelocity) {
                // f = ma
                double requiredAcceleration = verticalVelocity - targetVelocity;
                double force = totalMass * requiredAcceleration;
                ans[0] = force / MAIN_ENGINE_MAX_FORCE;
                ans[1] = (force - MAIN_ENGINE_MAX_FORCE) / (BALANCE_ENGINES_MAX_FORCE * 2); // fine if negative
            }

            return ans;
        }

        double[] engineForcesLandingWhenAngleZero(double velocityAtSurface) {
            double[] ans = new double[2];

            // a = (v_final^2 - v^2) / 2h
            double requiredAcc = -(Math.pow(velocityAtSurface, 2) - Math.pow(verticalVelocity, 2)) / (2 * altitude);
            double moonAcc = Moon.getVerticalAccelerationOnObject(horizontalVelocity, altitude);
            // f = ma
            double requiredForce = totalMass * (moonAcc + requiredAcc);
            ans[0] = requiredForce / MAIN_ENGINE_MAX_FORCE;
            ans[1] = (requiredForce - MAIN_ENGINE_MAX_FORCE) / (BALANCE_ENGINES_MAX_FORCE * 2); // fine if negative

            return ans;
        }
    }

    public static void main(String[] args) {
        Spacecraft spacecraft = new Spacecraft();
        double time = 0;

        // ***** Main simulation loop ******
        enum States {SET_ANGLE_1, SET_H_VELOCITY, SET_ANGLE_2, SET_V_VELOCITY, NOTHING}
        States state = States.SET_ANGLE_1;
        double angle1 = -63, angle2 = 0, hVelocityReq = 2, vVelocityReq = 2;

        System.out.println("Simulating Bereshit's Landing:");
        System.out.println(spacecraft);
        while (true) {
            if (time % 100 == 0 || spacecraft.altitude < 100) {
                System.out.println(spacecraft);
            }
            if (spacecraft.altitude < 0) break;

            double[] balanceEnginesOutput = new double[2]; // normalized
            double mainEngineOutput = 0; // normalized
            double[] reqEnginesForces = new double[3]; // normalized

            if (state == States.SET_ANGLE_1) {
//                if (Math.abs(spacecraft.angle - angle1) > EPSILON || Math.abs(spacecraft.spinVelocity) > EPSILON) {
//                    balanceEnginesOutput = spacecraft.engineForcesForAngle(angle1);
//                }
//                reqEnginesForces = spacecraft.engineForcesForTarget(angle1, true);
                balanceEnginesOutput = spacecraft.balanceEnginesForcesForTarget(angle1);
            }
            else if (state == States.SET_H_VELOCITY) {
                if (spacecraft.horizontalVelocity > hVelocityReq) {
                    mainEngineOutput = 1;
                    balanceEnginesOutput = new double[]{1, 1};
                }
                reqEnginesForces = spacecraft.engineForcesForTarget(hVelocityReq, false);

            }
            else if (state == States.SET_ANGLE_2) {
//                balanceEnginesOutput = spacecraft.engineForcesForAngle(angle2);
//                reqEnginesForces = spacecraft.engineForcesForTarget(angle2, true);
                balanceEnginesOutput = spacecraft.balanceEnginesForcesForTarget(angle2);
            }
            else if (state == States.SET_V_VELOCITY) {
//                double[] tmp = spacecraft.engineForcesForVerticalVelocityWhenAngleZero(velocity2);
//                mainEngineOutput = tmp[0];
//                balanceEnginesOutput[0] = tmp[1];
//                balanceEnginesOutput[1] = tmp[1];

//                double[] tmp = spacecraft.engineForcesLandingWhenAngleZero(velocity2);
//                mainEngineOutput = tmp[0];
//                balanceEnginesOutput[0] = tmp[1];
//                balanceEnginesOutput[1] = tmp[1];
                reqEnginesForces = spacecraft.engineForcesForTarget(vVelocityReq, false);
                mainEngineOutput = spacecraft.mainEngineForceForLanding();
                if (mainEngineOutput > MAIN_ENGINE_MAX_FORCE) {
                    double balanceOutputs = (mainEngineOutput - MAIN_ENGINE_MAX_FORCE) / 2;
                    balanceEnginesOutput[0] = balanceOutputs;
                    balanceEnginesOutput[1] = balanceOutputs;
                }
            }
            spacecraft.applyEnginesNormalizedForces(mainEngineOutput, balanceEnginesOutput[0], balanceEnginesOutput[1]);
            spacecraft.update();
//            spacecraft.applyEnginesForces(reqEnginesForces[0], reqEnginesForces[1], reqEnginesForces[2]);

            if (state == States.SET_ANGLE_1 &&
                    Math.abs(spacecraft.spinVelocity) < EPSILON &&
                    Math.abs(spacecraft.angle - angle1) < EPSILON) {
                state = States.SET_H_VELOCITY;
            }
            else if (state == States.SET_H_VELOCITY &&
                    spacecraft.horizontalVelocity <= hVelocityReq) {
                state = States.SET_ANGLE_2;
            }
            else if (state == States.SET_ANGLE_2 && Math.abs(spacecraft.spinVelocity) < EPSILON) {
                state = States.SET_V_VELOCITY;
            }
//            else if (state == States.SET_V_VELOCITY) {
//                if (spacecraft.verticalVelocity <= vVelocityReq) {
//                    state = States.NOTHING;
//                }
//            }
//            if (state == States.NOTHING && spacecraft.verticalVelocity > vVelocityReq) {
//                state = States.SET_V_VELOCITY;
//            }

            time += DT;
        }
    }
}