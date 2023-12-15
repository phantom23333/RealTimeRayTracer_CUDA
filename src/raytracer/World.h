#pragma once

#include "Hittable.h"
#include "AABB.h"

#include "thrust/device_ptr.h"
#include "thrust/device_malloc.h"
#include "thrust/device_free.h"

__device__ inline AABB surrounding_box(AABB box0, AABB box1) {
    glm::vec3 Small(fmin(box0.min().x, box1.min().x),
        fmin(box0.min().y, box1.min().y),
        fmin(box0.min().z, box1.min().z));

    glm::vec3 big(fmax(box0.max().x, box1.max().x),
        fmax(box0.max().y, box1.max().y),
        fmax(box0.max().z, box1.max().z));

    return AABB(Small, big);
}

class World : public Hittable {
public:
    int number_of_objects;
    int capacity;
    Hittable** objects;
    
    __device__ World() {
        objects = new Hittable*[20];
        number_of_objects = 0;
        capacity = 20;
    }

    __device__ World(int capacity) {
        objects = new Hittable*[capacity];
        number_of_objects = 0;
        capacity = capacity;
    }


    __device__ virtual bool hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const {
        HitRecord temp_rec;
        bool hit_anything = false;
        float closest_so_far = t_max;
        for (int i = 0; i < number_of_objects; i++) {
            if (objects[i]->hit(r, t_min, closest_so_far, temp_rec)) {
                hit_anything = true;
                closest_so_far = temp_rec.t;
                rec = temp_rec;
            }
        }
        return hit_anything;
    }

    __device__ bool add(Hittable* object) {
        if (number_of_objects >= capacity) {
			return false;
		}

        objects[number_of_objects] = object;
		number_of_objects++;
		return true;
    }

    __device__ bool bounding_box(float time0, float time1, AABB& output_box) const {
        if (number_of_objects < 1) {
            return false;
        }

        AABB temp_box;
        bool first_box = true;

        for (int i = 0; i < number_of_objects; i++) {
            if (!objects[i]->bounding_box(time0, time1, temp_box)) {
                return false;
            }
            output_box = first_box ? temp_box : surrounding_box(output_box, temp_box);
            first_box = false;
        }

        return true;
    }


    __device__ ~World() {
        for (int i = 0; i < number_of_objects; i++) {
            delete objects[i];
        }
        delete[] objects;
    }
};






