package com.polimi.comsol.util;

public class PhysicParam {

    // Class to hold physics data
    public static class Physics {
        public double mass;
        public double friction;
        public double restitution;
        public boolean movable;

        // Constructor
        public Physics() {
            this.mass = 1.0;
            this.friction = 0.5;
            this.restitution = 0.5;
            this.movable = true;
        }

        // Method to set physics properties from a JSON object
        public void setPhysics(JSONObject physicsJson) throws JSONException {
            if (physicsJson.has("mass")) {
                this.mass = physicsJson.getDouble("mass");
            }
            if (physicsJson.has("friction")) {
                this.friction = physicsJson.getDouble("friction");
            }
            if (physicsJson.has("restitution")) {
                this.restitution = physicsJson.getDouble("restitution");
            }
            if (physicsJson.has("movable")) {
                this.movable = physicsJson.getBoolean("movable");
            }
        }
    }

    private Map<String, Geometry> geometries = new HashMap<>();
    private Map<String, Physics> physicsData = new HashMap<>();

    // Method to set geometry and physics from a JSON string
   public void setModelProperties(String modelJsonString) {
        try {
            JSONObject modelJson = new JSONObject(modelJsonString);

            // Process geometry data
            if (modelJson.has("geometry")) {
                JSONObject geometryJson = modelJson.getJSONObject("geometry");
                String geometryName = geometryJson.optString("name", "defaultGeometry"); // Default name if not provided
                Geometry geometry = new Geometry();
                geometry.setGeometry(geometryJson); // Set geometry properties
                geometries.put(geometryName, geometry);
            }

            // Process physics data
            if (modelJson.has("physics")) {
                JSONObject physicsJson = modelJson.getJSONObject("physics");
                String physicsName = modelJson.optString("name", "defaultPhysics");  // Default name if not provided.
                Physics physics = new Physics();
                physics.setPhysics(physicsJson); // Set physics properties
                physicsData.put(physicsName, physics);
            }

        } catch (JSONException e) {
            System.err.println("Error parsing JSON: " + e.getMessage());
            // Consider throwing the exception or using a more sophisticated error handling mechanism.
            //  For example:
            //  throw new IllegalArgumentException("Invalid JSON for model properties", e);
        }
    }

}
