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

import boofcv.abst.geo.TriangulateTwoViewsCalibrated;
import boofcv.alg.geo.PositiveDepthConstraintCheck;
import boofcv.factory.geo.FactoryMultiView;
import boofcv.struct.geo.AssociatedPair;
import georegression.struct.se.Se3_F64;

import java.util.List;

/**
 * Given a set of observations in normalized image coordinates and a set of possible
 * stereo transforms select the best view
 *
 * @author Peter Abeles
 */
public class SelectBestStereoTransform {

	// used to select best hypothesis
	PositiveDepthConstraintCheck depthCheck;

	/**
	 * Specifies how the essential matrix is computed
	 *
	 */
	public SelectBestStereoTransform(TriangulateTwoViewsCalibrated triangulate ) {
		this.depthCheck = new PositiveDepthConstraintCheck(triangulate);
	}

	public SelectBestStereoTransform() {
		this(FactoryMultiView.triangulateTwoGeometric() );
	}

	public void select(List<Se3_F64> candidatesAtoB,
					   List<AssociatedPair> observations ,
					   Se3_F64 model ) {

		// use positive depth constraint to select the best one
		Se3_F64 bestModel = null;
		int bestCount = -1;
		for( int i = 0; i < candidatesAtoB.size(); i++ ) {
			Se3_F64 s = candidatesAtoB.get(i);
			int count = 0;
			for( AssociatedPair p : observations ) {
				if( depthCheck.checkConstraint(p.p1,p.p2,s)) {
					count++;
				}
			}

			if( count > bestCount ) {
				bestCount = count;
				bestModel = s;
			}
		}

		if( bestModel == null )
			throw new RuntimeException("BUG");

		model.set(bestModel);
	}

}