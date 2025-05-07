package com.polimi.comsol.util;

import org.json.JSONObject;
import org.json.JSONArray;
import org.json.JSONException;

public class GeometricParam {
    // Class to hold geometry data
        public String type;
        public double[] dimensions; // For various shapes
        public double radius; //For sphere
        public double height; //For cylinder
        public double width;  //for box
        public double depth; //for box
        public double x,y,z; //position
        public int radialSegments; // For Cylinder
        public int heightSegments;   // For Cylinder
        public int widthSegments; // For Box
        public int depthSegments; // For Box

        // Constructor
        public GeometricParam() {
            this.type = "box"; // Default type
            this.dimensions = new double[]{1, 1, 1}; // Default dimensions for a box
            this.radius = 1;
            this.height = 1;
            this.width = 1;
            this.depth = 1;
            this.x = 0;
            this.y = 0;
            this.z = 0;
            this.radialSegments = 32;
            this.heightSegments = 1;
            this.widthSegments = 1;
            this.depthSegments = 1;
        }

        // Method to set geometry properties from a JSON object
        public void setGeometry(JSONObject geometryJson) throws JSONException {
            if (geometryJson.has("type")) {
                this.type = geometryJson.getString("type");
            }
            if (geometryJson.has("dimensions")) {
                JSONArray dimArray = geometryJson.getJSONArray("dimensions");
                if (dimArray.length() == 3) {
                    this.dimensions = new double[]{dimArray.getDouble(0), dimArray.getDouble(1), dimArray.getDouble(2)};
                }
            }
            if(geometryJson.has("radius")){
                this.radius = geometryJson.getDouble("radius");
            }
            if(geometryJson.has("height")){
                this.height = geometryJson.getDouble("height");
            }
            if(geometryJson.has("width")){
                this.width = geometryJson.getDouble("width");
            }
             if(geometryJson.has("depth")){
                this.depth = geometryJson.getDouble("depth");
            }
            if (geometryJson.has("x")) {
                this.x = geometryJson.getDouble("x");
            }
            if (geometryJson.has("y")) {
                this.y = geometryJson.getDouble("y");
            }
            if (geometryJson.has("z")) {
                this.z = geometryJson.getDouble("z");
            }
            if(geometryJson.has("radialSegments")){
                this.radialSegments = geometryJson.getInt("radialSegments");
            }
            if(geometryJson.has("heightSegments")){
                this.heightSegments = geometryJson.getInt("heightSegments");
            }
            if(geometryJson.has("widthSegments")){
                this.widthSegments = geometryJson.getInt("widthSegments");
            }
            if(geometryJson.has("depthSegments")){
                this.depthSegments = geometryJson.getInt("depthSegments");
            }

        }
    }

}
