/**
* This file is part of LSD-SLAM.
*
* Copyright 2013 Jakob Engel <engelj at in dot tum dot de> (Technical University of Munich)
* For more information see <http://vision.in.tum.de/lsdslam> 
*
* LSD-SLAM is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* LSD-SLAM is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with LSD-SLAM. If not, see <http://www.gnu.org/licenses/>.
*/

#include "GlobalMapping/KeyFrameGraph.h"
#include "DataStructures/Frame.h"

#include <g2o/core/sparse_optimizer.h>
#include <g2o/solvers/pcg/linear_solver_pcg.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/solver.h>
#include <g2o/core/optimization_algorithm_dogleg.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/estimate_propagator.h>
#include <g2o/core/sparse_optimizer_terminate_action.h>

#include "opencv2/opencv.hpp"

#include <g2o/types/sim3/sim3.h>
#include "GlobalMapping/g2oTypeSim3Sophus.h"


#include "IOWrapper/ImageDisplay.h"

// for mkdir
#include <sys/types.h>
#include <sys/stat.h>
// for iterating over files in a directory
#include <dirent.h>
#include <queue>

#include <iostream>
#include <fstream>

#include "util/globalFuncs.h"

namespace lsd_slam
{


KFConstraintStruct::~KFConstraintStruct()
{
	if(edge != 0)
		delete edge;
#ifdef USE_GTSAM_OPT
		//delete betweenFactor; // DEBUG
#endif
}

/*
#ifdef USE_GTSAM_OPT
KFConstraintStructGtsam::~KFConstraintStructGtsam()
{
	if(betweenFactor != 0)
		delete betweenFactor;
}
#endif
*/


KeyFrameGraph::KeyFrameGraph()
: nextEdgeId(0)
{
	typedef g2o::BlockSolver_7_3 BlockSolver;
	typedef g2o::LinearSolverCSparse<BlockSolver::PoseMatrixType> LinearSolver;
	//typedef g2o::LinearSolverPCG<BlockSolver::PoseMatrixType> LinearSolver;
	LinearSolver* solver = new LinearSolver();
	BlockSolver* blockSolver = new BlockSolver(solver);
	g2o::OptimizationAlgorithmLevenberg* algorithm = new g2o::OptimizationAlgorithmLevenberg(blockSolver);
	graph.setAlgorithm(algorithm);
	
    graph.setVerbose(false); // printOptimizationInfo
	solver->setWriteDebug(true);
	blockSolver->setWriteDebug(true);
	algorithm->setWriteDebug(true);

#ifdef USE_GTSAM_OPT
	//optimizerGtsam(graphGtsam, initialEstimateGtsam);
	//marginalsGtsam(graphGtsam, resultGtsam); //TODO same as above. Goes to information somehow
#endif



	totalPoints=0;
	totalEdges=0;
	totalVertices=0;


}

KeyFrameGraph::~KeyFrameGraph()
{
	// deletes edges
	for (KFConstraintStruct* edge : newEdgeBuffer)
		delete edge;	// deletes the g2oedge, which deletes the kernel.

/*
#ifdef USE_GTSAM_OPT
	// deletes edges
	for (KFConstraintStructGtsam* edge : newEdgeBufferGtsam)
		delete edge;	// deletes the gtsamedge, which deletes the kernel.
#endif
*/
	// deletes keyframes (by deleting respective shared pointers).

	idToKeyFrame.clear();

	// deletes pose structs (their memory is managed by graph)
	// WARNING: at this point, all Frames have to be deleted, otherwise it night cause segfaults!
	for(FramePoseStruct* p : allFramePoses)
		delete p;
}


void KeyFrameGraph::addFrame(Frame* frame)
{

	frame->pose->isRegisteredToGraph = true;
	FramePoseStruct* pose = frame->pose;


	allFramePosesMutex.lock();
	allFramePoses.push_back(pose);
	allFramePosesMutex.unlock();
}

void KeyFrameGraph::dumpMap(std::string folder)
{
	printf("DUMP MAP: dumping to %s\n", folder.c_str());

	keyframesAllMutex.lock_shared();
	char buf[100];
	int succ = system(("rm -rf "+folder).c_str());
	succ += system(("mkdir "+folder).c_str());

	for(unsigned int i=0;i<keyframesAll.size();i++)
	{
		snprintf(buf, 100, "%s/depth-%d.png", folder.c_str(), i);
		cv::imwrite(buf, getDepthRainbowPlot(keyframesAll[i], 0));

		snprintf(buf, 100, "%s/frame-%d.png", folder.c_str(), i);
		cv::imwrite(buf, cv::Mat(keyframesAll[i]->height(), keyframesAll[i]->width(),CV_32F,keyframesAll[i]->image()));

		snprintf(buf, 100, "%s/var-%d.png", folder.c_str(), i);
		cv::imwrite(buf, getVarRedGreenPlot(keyframesAll[i]->idepthVar(),keyframesAll[i]->image(),keyframesAll[i]->width(),keyframesAll[i]->height()));
	}


	int i = keyframesAll.size()-1;
	Util::displayImage("VAR PREVIEW", getVarRedGreenPlot(keyframesAll[i]->idepthVar(),keyframesAll[i]->image(),keyframesAll[i]->width(),keyframesAll[i]->height()));

	printf("DUMP MAP (succ %d): dumped %d depthmaps\n", succ,  (int)keyframesAll.size());

	Eigen::MatrixXf res, resD, resP, huber, usage, consistency, distance, error;
	Eigen::VectorXf meanRootInformation, usedPixels;

	res.resize(keyframesAll.size(), keyframesAll.size());
	resD.resize(keyframesAll.size(), keyframesAll.size());
	resP.resize(keyframesAll.size(), keyframesAll.size());
	usage.resize(keyframesAll.size(), keyframesAll.size());
	consistency.resize(keyframesAll.size(), keyframesAll.size());
	distance.resize(keyframesAll.size(), keyframesAll.size());
	error.resize(keyframesAll.size(), keyframesAll.size());
	meanRootInformation.resize(keyframesAll.size());
	usedPixels.resize(keyframesAll.size());
	res.setZero();
	resD.setZero();
	resP.setZero();
	usage.setZero();
	consistency.setZero();
	distance.setZero();
	error.setZero();
	meanRootInformation.setZero();
	usedPixels.setZero();

	for(unsigned int i=0;i<keyframesAll.size();i++)
	{
		meanRootInformation[i] = keyframesAll[i]->meanInformation;
		usedPixels[i] = keyframesAll[i]->numPoints / (float)keyframesAll[i]->numMappablePixels;
	}

	// TODO for gtsam
	edgesListsMutex.lock_shared();
	for(unsigned int i=0;i<edgesAll.size();i++)
	{
		KFConstraintStruct* e = edgesAll[i];

		res(e->firstFrame->idxInKeyframes, e->secondFrame->idxInKeyframes) = e->meanResidual;
		resD(e->firstFrame->idxInKeyframes, e->secondFrame->idxInKeyframes) = e->meanResidualD;
		resP(e->firstFrame->idxInKeyframes, e->secondFrame->idxInKeyframes) = e->meanResidualP;
		usage(e->firstFrame->idxInKeyframes, e->secondFrame->idxInKeyframes) = e->usage;
		consistency(e->firstFrame->idxInKeyframes, e->secondFrame->idxInKeyframes) = e->reciprocalConsistency;
		distance(e->firstFrame->idxInKeyframes, e->secondFrame->idxInKeyframes) = e->secondToFirst.translation().norm();
		error(e->firstFrame->idxInKeyframes, e->secondFrame->idxInKeyframes) = e->edge->chi2();
	}
	edgesListsMutex.unlock_shared();
	keyframesAllMutex.unlock_shared();


	std::ofstream fle;

	fle.open(folder+"/residual.txt");
	fle << res;
	fle.close();

	fle.open(folder+"/residualD.txt");
	fle << resD;
	fle.close();

	fle.open(folder+"/residualP.txt");
	fle << resP;
	fle.close();

	fle.open(folder+"/usage.txt");
	fle << usage;
	fle.close();

	fle.open(folder+"/consistency.txt");
	fle << consistency;
	fle.close();

	fle.open(folder+"/distance.txt");
	fle << distance;
	fle.close();

	fle.open(folder+"/error.txt");
	fle << error;
	fle.close();

	fle.open(folder+"/meanRootInformation.txt");
	fle << meanRootInformation;
	fle.close();

	fle.open(folder+"/usedPixels.txt");
	fle << usedPixels;
	fle.close();

	printf("DUMP MAP: dumped %d edges\n", (int)edgesAll.size());
}



void KeyFrameGraph::addKeyFrame(Frame* frame)
{
	if(frame->pose->graphVertex != nullptr)
		return;

	// Insert vertex into g2o graph
	VertexSim3* vertex = new VertexSim3();
	vertex->setId(frame->id());

	Sophus::Sim3d camToWorld_estimate = frame->getScaledCamToWorld();

	if(!frame->hasTrackingParent())
		vertex->setFixed(true);  //  TODO: How to manage with GTSAM ?

	vertex->setEstimate(camToWorld_estimate);
	vertex->setMarginalized(false);

#ifdef USE_GTSAM_OPT
	initialEstimateGtsam.insert(frame->id(),moses3FromSim3(camToWorld_estimate)); // Notice frameID are increasing, but not consicutive!
#endif

	frame->pose->graphVertex = vertex;

	newKeyframesBuffer.push_back(frame);

}

void KeyFrameGraph::insertConstraint(KFConstraintStruct* constraint)
{
	EdgeSim3* edge = new EdgeSim3();
	edge->setId(nextEdgeId);
	++ nextEdgeId;

	totalEdges++;

	edge->setMeasurement(constraint->secondToFirst);
	edge->setInformation(constraint->information);
	edge->setRobustKernel(constraint->robustKernel);

	edge->resize(2);
	assert(constraint->firstFrame->pose->graphVertex != nullptr);
	edge->setVertex(0, constraint->firstFrame->pose->graphVertex);
	assert(constraint->secondFrame->pose->graphVertex != nullptr);
	edge->setVertex(1, constraint->secondFrame->pose->graphVertex);

#ifdef USE_GTSAM_OPT
	// DEBUG TEMP CONST NOISE MODEL
	//gtsam::noiseModel::Diagonal::shared_ptr odometryNoise = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(7) << 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1));
	
	//Create between factor with noise model and add to constraint
	//constraint->betweenFactor = new gtsam::BetweenFactor<gtsam::Moses3>(constraint->firstFrame->id(),constraint->secondFrame->id(),moses3FromSim3(constraint->secondToFirst),odometryNoise);	
#endif

	constraint->edge = edge;
	newEdgeBuffer.push_back(constraint);


	constraint->firstFrame->neighbors.insert(constraint->secondFrame);
	constraint->secondFrame->neighbors.insert(constraint->firstFrame);

	for(int i=0;i<totalVertices;i++)
	{
		//shortestDistancesMap
	}



	edgesListsMutex.lock();
	constraint->idxInAllEdges = edgesAll.size();
	edgesAll.push_back(constraint);
	edgesListsMutex.unlock();
}


/*
#ifdef USE_GTSAM_OPT
void KeyFrameGraph::insertConstraint(KFConstraintStructGtsam* constraint)
{
	totalEdgesGtsam++;

	//edge->setMeasurement(constraint->secondToFirst);
	//edge->setInformation(constraint->information);
	//edge->setRobustKernel(constraint->robustKernel);

	assert(constraint->firstFrame->pose->graphVertex != nullptr);
	assert(constraint->secondFrame->pose->graphVertex != nullptr);

	// DEBUG TEMP CONST NOISE MODEL
	gtsam::noiseModel::Diagonal::shared_ptr odometryNoise = noiseModel::Diagonal::Sigmas((Vector(7) << 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1));

	constraint->betweenFactor = new gtsam::BetweenFactor<Moses3>(constraint->firstFrame->id(),constraint->secondFrame->id(),moses3FromSim3(constraint->secondToFirst),odometryNoise);
	newEdgeBufferGtsam.push_back(constraint);

	// MODIFIES FRAMES (Also happening in parallel in g2o constrain fn above. Should happen in only one, hence commented below) TODO DEBUG
	//constraint->firstFrame->neighbors.insert(constraint->secondFrame);
	//constraint->secondFrame->neighbors.insert(constraint->firstFrame);

	for(int i=0;i<totalVertices;i++)
	{
		//shortestDistancesMap
	}



	edgesListsMutexGtsam.lock();
	constraint->idxInAllEdges = edgesAllGtsam.size();
	edgesAllGtsam.push_back(constraint);
	edgesListsMutexGtsam.unlock();
}
#endif
*/


bool KeyFrameGraph::addElementsFromBuffer()
{
	bool added = false; //WTF is this doing ?

#ifdef USE_GTSAM_OPT
	//DEBUG BIG TODO TAK needs to be in insert constraint with original information matrix
	gtsam::noiseModel::Diagonal::shared_ptr odometryNoise = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(7) << 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1));
#endif	

	keyframesForRetrackMutex.lock();
	for (auto newKF : newKeyframesBuffer)
	{
		graph.addVertex(newKF->pose->graphVertex);
		assert(!newKF->pose->isInGraph);
		newKF->pose->isInGraph = true;

		keyframesForRetrack.push_back(newKF);

		added = true;
	}
	keyframesForRetrackMutex.unlock();

	newKeyframesBuffer.clear();
	for (auto edge : newEdgeBuffer)
	{
		graph.addEdge(edge->edge);
#ifdef USE_GTSAM_OPT
		//Add BetweenFactors
		graphGtsam.add(gtsam::BetweenFactor<gtsam::Moses3>(edge->firstFrame->id(),edge->secondFrame->id(),moses3FromSim3(edge->secondToFirst),odometryNoise));
#endif	

		added = true;
	}
	newEdgeBuffer.clear();

/*
	//Add BetweenFactors
#ifdef USE_GTSAM_OPT
	for (auto edge : newEdgeBufferGtsam)
	{
		graphGtsam.add(edge->betweenFactor);
		added = true;
	}
	newEdgeBufferGtsam.clear();
#endif
*/
	return added;
}

int KeyFrameGraph::optimize(int num_iterations)
{
	// Abort if graph is empty, g2o shows an error otherwise
	if (graph.edges().size() == 0)
		return 0;
	
	graph.setVerbose(false); // printOptimizationInfo
	graph.initializeOptimization();
	
#ifdef USE_GTSAM_OPT
	//DEBUGresultGtsam = optimizerGtsam.optimize();
	gtsam::LevenbergMarquardtOptimizer optimizerGtsam(graphGtsam, initialEstimateGtsam);
	resultGtsam = optimizerGtsam.optimize();
	graphGtsam.print("Graph\n");
	gtsam::Marginals marginalsGtsam(graphGtsam, resultGtsam); //TODO same as above. Goes to information somehow
#endif

	return graph.optimize(num_iterations, false);

}



void KeyFrameGraph::calculateGraphDistancesToFrame(Frame* startFrame, std::unordered_map< Frame*, int >* distanceMap)
{
	distanceMap->insert(std::make_pair(startFrame, 0));
	
	std::multimap< int, Frame* > priorityQueue;
	priorityQueue.insert(std::make_pair(0, startFrame));
	while (! priorityQueue.empty())
	{
		auto it = priorityQueue.begin();
		int length = it->first;
		Frame* frame = it->second;
		priorityQueue.erase(it);
		
		auto mapEntry = distanceMap->find(frame);
		
		if (mapEntry != distanceMap->end() && length > mapEntry->second)
		{
			continue;
		}
		
		for (Frame* neighbor : frame->neighbors)
		{
			auto neighborMapEntry = distanceMap->find(neighbor);
			
			if (neighborMapEntry != distanceMap->end() && length + 1 >= neighborMapEntry->second)
				continue;
			
			if (neighborMapEntry != distanceMap->end())
				neighborMapEntry->second = length + 1;
			else
				distanceMap->insert(std::make_pair(neighbor, length + 1));
			priorityQueue.insert(std::make_pair(length + 1, neighbor));
		}
	}
}

}
