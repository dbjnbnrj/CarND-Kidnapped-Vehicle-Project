/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */


#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

// declare a random engine to be used across multiple and various method calls
static default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  default_random_engine gen;
  num_particles = 101;

  // define normal distributions for sensor noise
  normal_distribution <double> dist_x(0, std[0]);
  normal_distribution <double> dist_y(0, std[1]);
  normal_distribution <double> dist_theta(0, std[2]);

  for (int i = 0; i <num_particles; i++) {
    Particle p;
    p.id = i;
    p.x = x + dist_x(gen);
    p.y = y + dist_y(gen);
    p.theta = theta + dist_theta(gen);
    p.weight = 1.0;
    particles.push_back(p);
  }
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
  // Add measurements to each particle and add random Gaussian noise.
  normal_distribution <double> dist_x(0, std_pos[0]);
  normal_distribution <double> dist_y(0, std_pos[1]);
  normal_distribution <double> dist_theta(0, std_pos[2]);

  for (int i = 0; i <num_particles; i++) {

    // calculate new state
    if (fabs(yaw_rate) <0.00001) {
      particles[i].x += velocity * delta_t * cos(particles[i].theta);
      particles[i].y += velocity * delta_t * sin(particles[i].theta);
    } else {
      particles[i].x += velocity / yaw_rate * (sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta));
      particles[i].y += velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate * delta_t));
      particles[i].theta += yaw_rate * delta_t;
    }

    // add noise
    particles[i].x += dist_x(gen);
    particles[i].y += dist_y(gen);
    particles[i].theta += dist_theta(gen);
  }

}

void ParticleFilter::dataAssociation(std::vector <LandmarkObs> predicted, std::vector <LandmarkObs> & observations) {
  for (int i = 0; i <observations.size(); i++) {
    LandmarkObs o = observations[i];
    double min_val = 99999.0;
    int map_id = -1;
    for (int j = 0; j <predicted.size(); j++) {
      LandmarkObs pred = predicted[j];
      double val = dist(o.x, o.y, pred.x, pred.y);
      if (val <min_val) {
        min_val = val;
        map_id = pred.id;
      }
    }
    observations[i].id = map_id;
  }
}

vector <LandmarkObs> ParticleFilter::calculatePredictions(const vector<Map::single_landmark_s>& landmarks, double sensor_range, Particle &p){
	vector <LandmarkObs> predictions;
	for (int i = 0; i <landmarks.size(); i++) {
		float x = landmarks[i].x_f;
		float y = landmarks[i].y_f;
		int id = landmarks[i].id_i;

		// As long as landmarks are in the radius of the sensor they are counted
		if (fabs(x - p.x) <= sensor_range && fabs(y - p.y) <= sensor_range) {
			predictions.push_back(LandmarkObs { id, x, y } );
		}
	}

	return predictions;
}

vector <LandmarkObs> ParticleFilter::convertCarToMapCoords(const std::vector<LandmarkObs>& car_obs, Particle& p){
    vector <LandmarkObs> map_obs;
    for (int i = 0; i <car_obs.size(); i++) {
      double x = cos(p.theta) * car_obs[i].x - sin(p.theta) * car_obs[i].y + p.x;
      double y = sin(p.theta) * car_obs[i].x + cos(p.theta) * car_obs[i].y + p.y;
      map_obs.push_back(LandmarkObs { car_obs[i].id, x, y });
    }
		return map_obs;
}

void ParticleFilter::updateParticleWeights(std::vector <LandmarkObs> & map_obs, std::vector <LandmarkObs> & landmarks, double std_landmark[], Particle & p) {
		p.weight = 1.0;
		for (int i = 0; i <map_obs.size(); i++) {
			double map_x = map_obs[i].x;
			double map_y = map_obs[i].y;
			int id = map_obs[i].id;

			double l_x, l_y;
			for (int j= 0; j<landmarks.size(); j++) {
				if (landmarks[j].id == id) {
					l_x = landmarks[j].x;
					l_y = landmarks[j].y;
					break;
				}
			}

			double s_x = std_landmark[0];
			double s_y = std_landmark[1];
			double weight = (1 / (2 * M_PI * s_x * s_y)) * exp(-(pow(l_x - map_x, 2) / (2 * pow(s_x, 2)) + (pow(l_y - map_y, 2) / (2 * pow(s_y, 2)))));
			p.weight *= weight;
		}
}


void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
  const std::vector <LandmarkObs> & observations,
    const Map & map_landmarks) {

  for (int i = 0; i <num_particles; i++) {
    double p_x = particles[i].x;
    double p_y = particles[i].y;
    double p_theta = particles[i].theta;

    vector <LandmarkObs> landmarks = calculatePredictions(map_landmarks.landmark_list, sensor_range, particles[i]);
    vector <LandmarkObs> map_obs = convertCarToMapCoords(observations, particles[i]);
		dataAssociation(landmarks, map_obs);
		updateParticleWeights(map_obs, landmarks, std_landmark, particles[i]);
  }
}

void ParticleFilter::resample() {
  vector <Particle> new_sample;
  vector <double> weights;
  for (int i = 0; i <num_particles; i++) {
    weights.push_back(particles[i].weight);
  }
  uniform_int_distribution <int> particle_dist(0, num_particles - 1);
  int index = particle_dist(gen);

  double max_weight = * max_element(weights.begin(), weights.end());
  uniform_real_distribution <double> weight_dist(0.0, max_weight);
  double beta = 0.0;
  for (int i = 0; i <num_particles; i++) {
    beta += weight_dist(gen) * 2.0;
		while (beta> weights[index]) {
      beta -= weights[index];
      index = (index + 1) % num_particles;
    }
    new_sample.push_back(particles[index]);
  }
  particles = new_sample;
}

Particle ParticleFilter::SetAssociations(Particle & particle,
  const std::vector <int> & associations,
    const std::vector <double> & sense_x,
      const std::vector <double> & sense_y) {
  //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates

  particle.associations = associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
	return particle;
}

string ParticleFilter::getAssociations(Particle best) {
  vector <int> v = best.associations;
  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator <int> (ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1); // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseX(Particle best) {
  vector <double> v = best.sense_x;
  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator <float> (ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1); // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseY(Particle best) {
  vector <double> v = best.sense_y;
  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator <float> (ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1); // get rid of the trailing space
  return s;
}
