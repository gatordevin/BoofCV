/*
 * Copyright (c) 2011-2018, Peter Abeles. All Rights Reserved.
 *
 * This file is part of BoofCV (http://boofcv.org).
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package boofcv.alg.geo.robust;

import boofcv.alg.geo.f.EpipolarMinimizeGeometricError;
import boofcv.struct.geo.AssociatedPair;
import org.ddogleg.fitting.modelset.DistanceFromModel;
import org.ejml.data.DMatrixRMaj;

import java.util.List;

/**
 * Computes geometric error in an uncalibrated stereo pair. This error is equivalent to
 * computing the optimal 3D triangulation, reprojecting the point, and computing the symmetric error.
 *
 * @see EpipolarMinimizeGeometricError
 *
 * @author Peter Abeles
 */
public class DistanceFundamentalGeometric implements DistanceFromModel<DMatrixRMaj, AssociatedPair> {

	EpipolarMinimizeGeometricError adjuster = new EpipolarMinimizeGeometricError();
	AssociatedPair adjusted = new AssociatedPair();

	DMatrixRMaj F21;

	@Override
	public void setModel(DMatrixRMaj model) {
		this.F21 = model;
	}

	@Override
	public double computeDistance(AssociatedPair original) {
		if( !adjuster.process(F21,original.p1.x,original.p1.y,original.p2.x,original.p2.y,
				adjusted.p1,adjusted.p2) ) {
			return Double.MAX_VALUE;
		}

		// Since the adjusted observations will intersect perfectly there's no need to triangulate
		// then reproject. This was verified empirically.
		return original.p1.distance2(adjusted.p1) + original.p2.distance2(adjusted.p2);
	}

	@Override
	public void computeDistance(List<AssociatedPair> associatedPairs, double[] distance) {
		for (int i = 0; i < associatedPairs.size(); i++) {
			distance[i] = computeDistance(associatedPairs.get(i));
		}
	}

	@Override
	public Class<AssociatedPair> getPointType() {
		return AssociatedPair.class;
	}

	@Override
	public Class<DMatrixRMaj> getModelType() {
		return DMatrixRMaj.class;
	}
}
