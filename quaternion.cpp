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
#include "quaternion.h"

#define M_PI		3.14159265358979323846f	/* pi */
#define M_PI_2		1.57079632679489661923f	/* pi/2 */

// Conversion factors
static constexpr float DEG_TO_RAD = M_PI / 180.0f;
static constexpr float RAD_TO_DEG = 180.0f / M_PI;
static constexpr float EPSILON = 1e-6f;

    void Quaternion::normalize() {
        float mag = sqrtf(w*w + x*x + y*y + z*z);
        if (mag > EPSILON) {
            float inv = 1.0f / mag;
            w *= inv; x *= inv; y *= inv; z *= inv;
        } else {
            w = 1.0f; x = y = z = 0.0f;
        }
    }
    
    float Quaternion::magnitude() const {
        return sqrtf(w*w + x*x + y*y + z*z);
    }
    
    Quaternion Quaternion::conjugate() const {
        return Quaternion(w, -x, -y, -z);
    }
    
    // Quaternion multiplication for relative rotation
    Quaternion Quaternion::operator*(const Quaternion& q) const {
        return Quaternion(
            w*q.w - x*q.x - y*q.y - z*q.z,  // w
            w*q.x + x*q.w + y*q.z - z*q.y,  // x
            w*q.y - x*q.z + y*q.w + z*q.x,  // y
            w*q.z + x*q.y - y*q.x + z*q.w   // z
        );
    }

    bool Quaternion::operator==(const Quaternion& other) const {
        return (fabsf(w - other.w) < EPSILON &&
                fabsf(x - other.x) < EPSILON &&
                fabsf(y - other.y) < EPSILON &&
                fabsf(z - other.z) < EPSILON);
    }

    bool Quaternion::operator!=(const Quaternion& other) const {
        return !(*this == other);
    }

namespace QuaternionMath {
    
    Quaternion average(const Quaternion* quats, uint32_t count) {
        if (count == 0) return Quaternion::identity();
        if (count == 1) return quats[0];
        
        Quaternion avg = quats[0];
        avg.normalize();
        
        // Iteratively refine the average
        for (int iter = 0; iter < 3; iter++) {
            Quaternion error_sum(0, 0, 0, 0);
            
            for (uint32_t i = 0; i < count; i++) {
                Quaternion q = quats[i];
                q.normalize();
                
                // Ensure shortest path by checking dot product
                float dot = avg.w*q.w + avg.x*q.x + avg.y*q.y + avg.z*q.z;
                if (dot < 0.0f) {
                    q.w = -q.w; q.x = -q.x; q.y = -q.y; q.z = -q.z;
                }
                
                // Calculate error quaternion: q * avg.conjugate()
                Quaternion error = q * avg.conjugate();
                
                // Convert to vector part (imaginary components)
                error_sum.x += error.x;
                error_sum.y += error.y;
                error_sum.z += error.z;
            }
            
            // Average the error vectors
            float inv_count = 1.0f / count;
            error_sum.x *= inv_count;
            error_sum.y *= inv_count;
            error_sum.z *= inv_count;
            
            // Convert back to quaternion and apply correction
            float error_mag = sqrtf(error_sum.x*error_sum.x + 
                                   error_sum.y*error_sum.y + 
                                   error_sum.z*error_sum.z);
            
            if (error_mag < 1e-6f) break; // Converged
            
            // Small angle approximation for the correction
            Quaternion correction(1.0f, error_sum.x, error_sum.y, error_sum.z);
            correction.normalize();
            
            avg = correction * avg;
            avg.normalize();
        }
        
        return avg;
    }
    
    Quaternion relativeRotation(const Quaternion& current, const Quaternion& reference) {
        Quaternion ref_conj = reference.conjugate();
        Quaternion relative = current * ref_conj;
        
        // Ensure shortest path: if w < 0, negate to get equivalent rotation with w > 0
        if (relative.w < 0.0f) {
            relative.w = -relative.w;
            relative.x = -relative.x;
            relative.y = -relative.y;
            relative.z = -relative.z;
        }
        
        return relative;
    }
    
    /**
     * @brief Convert quaternion to Euler angles (ZYX convention)
     * Returns angles in degrees
     */
    void toEulerAngles(const Quaternion& q, float& roll, float& pitch, float& yaw) {
        float norm = q.magnitude();
        float w = q.w / norm, x = q.x / norm, y = q.y / norm, z = q.z / norm;
        
        // float test = x*y + z*w;
        
        // if (test > 0.499f) { // Singularity at north pole
        //     yaw = 2.0f * atan2f(x, w) * RAD_TO_DEG;
        //     pitch = 90.0f;
        //     roll = 0.0f;
        //     return;
        // }
        
        // if (test < -0.499f) { // Singularity at south pole
        //     yaw = -2.0f * atan2f(x, w) * RAD_TO_DEG;
        //     pitch = -90.0f;
        //     roll = 0.0f;
        //     return;
        // }
        
        // float sqx = x*x, sqy = y*y, sqz = z*z;
        
        // yaw = atan2f(2.0f*y*w - 2.0f*x*z, 1.0f - 2.0f*sqy - 2.0f*sqz) * RAD_TO_DEG;
        // pitch = asinf(2.0f*test) * RAD_TO_DEG;
        // roll = atan2f(2.0f*x*w - 2.0f*y*z, 1.0f - 2.0f*sqx - 2.0f*sqz) * RAD_TO_DEG;
    // roll (x-axis rotation)
        float roll_rad, pitch_rad, yaw_rad;
        // roll
        float sinr_cosp = 2 * (w*x + y*z);
        float cosr_cosp = 1 - 2 * (x*x + y*y);
        roll_rad = atan2f(sinr_cosp,cosr_cosp);
        // pitch
        float sinp = sqrtf(1 + 2 * (w*y - x*z));
        float cosp = sqrtf(1 - 2 - (w*y + x*y));
        pitch_rad = 2 * atan2f(sinp,cosp) - M_PI / 2;
        // yaw
        float siny_cosp = 2 * (w*z + x*y);
        float cosy_cosp = 1 - 2 * (y*y + z*z);
        yaw_rad = atan2f(siny_cosp, cosy_cosp);

        roll = roll_rad * RAD_TO_DEG;
        pitch = pitch_rad * RAD_TO_DEG;
        yaw = yaw_rad * RAD_TO_DEG;

        // TODO: SEMANTICS:
        float temp = roll;
        roll = pitch;
        pitch = temp;
    }
    
    /**
     * @brief Get rotation angle magnitude from quaternion
     */
    float rotationAngle(const Quaternion& q) {
        float w_clamped = fabsf(q.w);
        if (w_clamped > 1.0f) w_clamped = 1.0f;
        return 2.0f * acosf(w_clamped) * RAD_TO_DEG;
    }

    /**
     * @brief Get distance angle between quaternions
     */
    float distance(const Quaternion& q1, const Quaternion& q2){
        float theta = acosf(q1.w*q2.w + q1.x*q2.x + q1.y*q2.y + q1.z*q2.z);
        if (theta>M_PI_2) theta = M_PI_2 - theta;
        return theta;
    }
    
}
