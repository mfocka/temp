/*
 ********************************************************************************
 *  ADB Safegate Belgium BV
 *  585 Leuvensesteenweg
 *  Zaventem 1930
 *
 *  This file is the intellectual property of ADB Safegate Belgium BV.
 *  This file shall not be duplicated, used, modified, or disclosed in whole or
 *  in part without the express written consent of ADB Safegate Belgium BV.
 *
 *  COPYRIGHT (C) ADB Safegate Belgium BV.
 *  ALL RIGHTS RESERVED.
 *
 *  Original Author: Mats Fockaert
 *  Original Date: 23 Jun 2025
 *  Project : EP195
 *  Processor: STM32H7
 *
 ********************************************************************************
 */

#ifndef QUATERNION_H
#define QUATERNION_H

#include <cstdint>
#include <cmath>

/**
 * @brief Quaternion structure for rotation representation
 * 
 * Quaternions are represented as q = w + xi + yj + zk where:
 * - w is the scalar (real) component
 * - x, y, z are the vector (imaginary) components
 * 
 * This matches ST Motion Library format: [w, x, y, z]
 */
struct Quaternion {
    float w;    // Scalar component (real part)
    float x;    // Vector i component (imaginary)
    float y;    // Vector j component (imaginary)
    float z;    // Vector k component (imaginary)
    
    // Constructors
    Quaternion() : w(1.0f), x(0.0f), y(0.0f), z(0.0f) {}
    Quaternion(float w_, float x_, float y_, float z_) : w(w_), x(x_), y(y_), z(z_) {}
    
    // // Copy constructor
    // Quaternion(const Quaternion& other) : w(other.w), x(other.x), y(other.y), z(other.z) {}
    
    // ST Motion Library integration (format: [W, X, Y, Z])
    void fromArray(const float* quat_array) {
        w = quat_array[0];
        x = quat_array[1];
        y = quat_array[2];
        z = quat_array[3];
    }
    
    void toArray(float* quat_array) const {
        quat_array[0] = w;
        quat_array[1] = x;
        quat_array[2] = y;
        quat_array[3] = z;
    }
    
    // Basic operations
    void normalize();
    float magnitude() const;
    Quaternion conjugate() const;
    // Quaternion inverse() const;
    
    // Static identity quaternion
    static Quaternion identity() {
        return Quaternion(1.0f, 0.0f, 0.0f, 0.0f);
    }
    
    // Validation
    bool isValid() const;
    bool isNormalized(float tolerance = 0.001f) const;
    
    // Conversion to rotation angle
    float toRotationAngle() const;
    
    // Operators
    // Quaternion operator+(const Quaternion& other) const;
    // Quaternion operator-(const Quaternion& other) const;
    Quaternion operator*(const Quaternion& other) const;  // Quaternion multiplication
    // Quaternion operator*(float scalar) const;             // Scalar multiplication
    bool operator==(const Quaternion& other) const;
    bool operator!=(const Quaternion& other) const;
    
    // Access by index (for compatibility)
    // float& operator[](int index);
    // const float& operator[](int index) const;
};

/**
 * @brief Quaternion utility functions for motion detection
 */
namespace QuaternionMath {
    
    /**
     * @brief Average multiple quaternions (useful for calibration)
     * @param quaternions Array of quaternions
     * @param count Number of quaternions
     * @return Averaged quaternion
     */
    Quaternion average(const Quaternion* quaternions, uint32_t count);
    
    /**
     * @brief Convert quaternion to Euler angles (roll, pitch, yaw) in degrees
     * @param q Input quaternion
     * @param roll Output roll angle in degrees
     * @param pitch Output pitch angle in degrees
     * @param yaw Output yaw angle in degrees
     */
    void toEulerAngles(const Quaternion& q, float& roll, float& pitch, float& yaw);
    
    /**
     * @brief Create quaternion from Euler angles (roll, pitch, yaw) in degrees
     * @param roll Roll angle in degrees
     * @param pitch Pitch angle in degrees
     * @param yaw Yaw angle in degrees
     * @return Quaternion representing the rotation
     */
    Quaternion fromEulerAngles(float roll, float pitch, float yaw);
    
    Quaternion relativeRotation(const Quaternion& current, const Quaternion& reference);

    float distance(const Quaternion& q1, const Quaternion& q2);
    // /**
    //  * @brief Create quaternion from axis-angle representation
    //  * @param axis_x X component of rotation axis (normalized)
    //  * @param axis_y Y component of rotation axis (normalized)
    //  * @param axis_z Z component of rotation axis (normalized)
    //  * @param angle_degrees Rotation angle in degrees
    //  * @return Quaternion representing the rotation
    //  */
    // Quaternion fromAxisAngle(float axis_x, float axis_y, float axis_z, float angle_degrees);
    
    // /**
    //  * @brief Extract axis-angle representation from quaternion
    //  * @param q Input quaternion
    //  * @param axis_x Output X component of rotation axis
    //  * @param axis_y Output Y component of rotation axis
    //  * @param axis_z Output Z component of rotation axis
    //  * @param angle_degrees Output rotation angle in degrees
    //  */
    // void toAxisAngle(const Quaternion& q, float& axis_x, float& axis_y, float& axis_z, float& angle_degrees);
    
    // /**
    //  * @brief Check if a quaternion is valid (no NaN or infinite values)
    //  * @param q Input quaternion
    //  * @return True if valid
    //  */
    // bool isValid(const Quaternion& q);
    
    // /**
    //  * @brief Check if a quaternion is normalized within tolerance
    //  * @param q Input quaternion
    //  * @param tolerance Tolerance for magnitude check
    //  * @return True if normalized
    //  */
    // bool isNormalized(const Quaternion& q, float tolerance = 0.001f);
    
    // /**
    //  * @brief Safe normalization that handles near-zero quaternions
    //  * @param q Input/output quaternion
    //  * @return True if normalization was successful
    //  */
    // bool safeNormalize(Quaternion& q);
    
    // Constants
    static constexpr float QUATERNION_TOLERANCE = 0.0001f;
    static const Quaternion IDENTITY;
}

#endif // QUATERNION_H