package Ex0;

public class Bereshit_101 {
    /* Spacecraft parameters */
    public static final double MASS_WHEN_EMPTY = 165; // kg
    public static final double MAIN_ENGINE_MAX_FORCE = 430; // Newton
    public static final double MAIN_ENGINE_MAX_FUEL_BURN = 0.15; // liters per sec

    static final int BALANCE_ENGINES_PER_SIDE = 4;
    public static final double BALANCE_ENGINES_MAX_FORCE = 25 * BALANCE_ENGINES_PER_SIDE; // Newton
    public static final double BALANCE_ENGINE_MAX_FUEL_BURN = 0.009 * BALANCE_ENGINES_PER_SIDE; // liters per sec
    public static final double BALANCE_ENGINES_MAX_ANGULAR_ACC = 1 * BALANCE_ENGINES_PER_SIDE; // degrees per sec^2

    /* Starting point parameters */
    public static final double INITIAL_ALTITUDE = 13748; // meters
    public static final double INITIAL_FUEL_MASS = 121.06; // liters
    public static final double INITIAL_VERTICAL_VELOCITY = 24.8; // m/s
    public static final double INITIAL_HORIZONTAL_VELOCITY = 932.2; // m/s
    public static final double INITIAL_ANGLE = 58.4; // 2-dimensional angle (degrees)
    public static final double INITIAL_DISTANCE = 181055; // meters
    public static final double DT = 1; // seconds

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
            verticalVelocity += Moon.ACCELERATION * DT;
        }

        public void applyEnginesForce(double mainPower, double frontPower, double backPower) {
            applyEnginesForce2(
                    mainPower / MAIN_ENGINE_MAX_FORCE,
                    frontPower / BALANCE_ENGINES_MAX_FORCE,
                    backPower / BALANCE_ENGINES_MAX_FORCE);
        }

        public void applyEnginesForce2(double mainNormalizedPower, double frontNormalizedPower, double backNormalizedPower) {
            // Normalize input powers
            mainNormalizedPower = Math.max(0, Math.min(1, mainNormalizedPower));
            frontNormalizedPower = Math.max(0, Math.min(1, frontNormalizedPower));
            backNormalizedPower = Math.max(0, Math.min(1, backNormalizedPower));

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
            altitude -= verticalVelocity * DT;
            angle += spinVelocity * DT;
            distance -= horizontalVelocity * DT;
            totalMass = MASS_WHEN_EMPTY + fuelMass;
        }

        double[] engineForcesForAngle(double targetAngle) {
            double[] forces = new double[2]; // [frontEngineForce, backEngineForce]
            // f = ma

            // Calculate front engine force
            if (spinVelocity >= 0 && angle > targetAngle) {
                double force = totalMass * (spinVelocity + Math.min(angle - targetAngle, BALANCE_ENGINES_MAX_ANGULAR_ACC));
                forces[0] = Math.min(force, BALANCE_ENGINES_MAX_FORCE);
            }

            // Calculate back engine force
            if (spinVelocity <= 0 && angle < targetAngle) {
                // f = ma
                double force = totalMass * (-spinVelocity + Math.min(targetAngle - angle, BALANCE_ENGINES_MAX_ANGULAR_ACC));
                forces[1] = Math.min(force, BALANCE_ENGINES_MAX_FORCE);
            }

            return forces;
        }
    }

    public static void main(String[] args) {
        Spacecraft spacecraft = new Spacecraft();
        double time = 0;

        // ***** Main simulation loop ******
        System.out.println("Simulating Bereshit's Landing:");
        while (spacecraft.altitude > 0) {
            if (time % 1 == 0 || spacecraft.altitude < 100) {
                System.out.println(spacecraft);
            }
            spacecraft.applyMoonGravity();

            double[] balanceEnginesOutput = new double[2]; // normalized
            double mainEngineOutput = 0; // normalized

            if (spacecraft.altitude > 2000) {
                balanceEnginesOutput = spacecraft.engineForcesForAngle(-50);
                if (Math.abs(spacecraft.verticalVelocity) > 1) {
//                    balanceEnginesOutput = spacecraft.calculateBalanceEnginesPowerForAngle(0);
                }
            }

            spacecraft.applyEnginesForce(mainEngineOutput, balanceEnginesOutput[0], balanceEnginesOutput[1]);
            spacecraft.update();
            time += DT;
        }
    }
}