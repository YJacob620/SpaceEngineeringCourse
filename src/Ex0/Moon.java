package Ex0;

public class Moon {
    public static final double RADIUS = 1737400; // meters
    public static final double MASS = 7.34767309 * Math.pow(10, 22); // kg
    public static final double GRAVITY_CONST = 1.622;// m/s^2
    public static final double EQ_SPEED = 1700;// m/s

    /**
     * Returns the vertical acceleration (downwards) the moon has on an
     * object at a give speed and altitude.
     *
     * @param speed    Object's speed
     * @param altitude Object's altitude from the surface of the moon
     * @return How much downward-acceleration the objects gets from the moon's gravity.
     */
    public static double getVerticalAccelerationOnObject(double speed, double altitude) {
        double totalRadius = RADIUS + altitude;
        double speedForNoEffect = Math.sqrt((GRAVITY_CONST * MASS) / totalRadius); // v = sqrt(GM/R)
//        double n = Math.abs(speed) / speedForNoEffect;
        double n = Math.abs(speed) / EQ_SPEED;
        return (1 - n) * GRAVITY_CONST;
    }

    public static double getForceOnObject(double totalMass) {
        // f = ma
        return totalMass * GRAVITY_CONST;
    }
}
