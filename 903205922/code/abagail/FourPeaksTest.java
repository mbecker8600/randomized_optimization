package opt.test;

import java.util.Arrays;

import dist.DiscreteDependencyTree;
import dist.DiscreteUniformDistribution;
import dist.Distribution;

import opt.DiscreteChangeOneNeighbor;
import opt.EvaluationFunction;
import opt.GenericHillClimbingProblem;
import opt.HillClimbingProblem;
import opt.NeighborFunction;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.example.*;
import opt.ga.*;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
import shared.FixedIterationTrainer;

/**
 * Copied from ContinuousPeaksTest
 * @version 1.0
 */
public class FourPeaksTest {
    /** The n value */
    private static final int[] N = {20, 40, 60, 80, 100};
    
    public static void main(String[] args) {

        double[][] rhcRuns = new double[5][5];
        double[][] saRuns = new double[5][5];
        double[][] gaRuns = new double[5][5];
        double[][] mimicRuns = new double[5][5];

        double[][] rhcTime = new double[5][5];
        double[][] saTime = new double[5][5];
        double[][] gaTime = new double[5][5];
        double[][] mimicTime = new double[5][5];

        double[][] rhcIterations = new double[5][5];
        double[][] saIterations = new double[5][5];
        double[][] gaIterations = new double[5][5];
        double[][] mimicIterations = new double[5][5];

        //hyperparameter initializations
        double[] saCooling = {.7, .8, .95};

        int[] gaPopSize = {100, 200, 400};
        CrossoverFunction[] gaCrossover = {new SingleCrossOver(), new TwoPointCrossOver(), new UniformCrossOver()};
        int[] gaCrossoverRate = {10, 40, 60, 80, 100};
        int[] gaMutationRate = {1, 10, 20, 50, 80, 100};

        int[] mimicNumSamplesTaken = {40, 80, 140};
        double[] mimicNumSamplesKept = {.2, .5, .9};

        for(int i=0; i<5; i++){
            for(int j=0; j<5; j++) {
                int[] ranges = new int[N[i]];
                Arrays.fill(ranges, 2);
                EvaluationFunction ef = new FourPeaksEvaluationFunction(N[i] / 5);
                Distribution odd = new DiscreteUniformDistribution(ranges);
                NeighborFunction nf = new DiscreteChangeOneNeighbor(ranges);
                MutationFunction mf = new DiscreteChangeOneMutation(ranges);
                Distribution df = new DiscreteDependencyTree(.1, ranges);
                HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
                ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);

                long starttime = System.currentTimeMillis();
                RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);
                FixedIterationTrainer fit = new FixedIterationTrainer(rhc, 200000);
                rhcIterations[j][i] = fit.train();
                double optimalRhc = ef.value(rhc.getOptimal());
                rhcRuns[j][i] = optimalRhc;
                System.out.println("RHC: " + optimalRhc);
                rhcTime[j][i] = System.currentTimeMillis() - starttime;

                double optimalSa = 0.0;
                for(double cooling : saCooling){
                    starttime = System.currentTimeMillis();
                    SimulatedAnnealing sa = new SimulatedAnnealing(1E11, cooling, hcp);
                    fit = new FixedIterationTrainer(sa, 200000);
                    double iterations = fit.train();
                    if(Math.max(ef.value(sa.getOptimal()), optimalSa) > optimalSa) {
                        saIterations[j][i] = iterations;
                        saTime[j][i] = System.currentTimeMillis() - starttime;
                    }
                    optimalSa = Math.max(ef.value(sa.getOptimal()), optimalSa);
                }
                saRuns[j][i] = optimalSa;
                System.out.println("SA: " + optimalSa);

                double optimalGa = 0.0;
                for(int popSize : gaPopSize){
                    for(CrossoverFunction crossover : gaCrossover){
                        for(int crossoverRate : gaCrossoverRate){
                            for(int mutationRate : gaMutationRate){
                                starttime = System.currentTimeMillis();
                                GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, crossover);
                                StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(popSize, crossoverRate, mutationRate, gap);
                                fit = new FixedIterationTrainer(ga, 1000);
                                double iterations = fit.train();
                                if(Math.max(ef.value(ga.getOptimal()), optimalGa) > optimalGa) {
                                    gaIterations[j][i] = iterations;
                                    gaTime[j][i] = System.currentTimeMillis() - starttime;
                                }
                                optimalGa = Math.max(ef.value(ga.getOptimal()), optimalGa);
                            }
                        }
                    }
                }
                gaRuns[j][i] = optimalGa;
                System.out.println("GA: " + optimalGa);

                double optimalMimic = 0.0;
                for(int sampleTaken: mimicNumSamplesTaken){
                    for(double sampleKept: mimicNumSamplesKept){
                        starttime = System.currentTimeMillis();
                        int sampleToKeep = Double.valueOf(sampleTaken*sampleKept).intValue();
                        MIMIC mimic = new MIMIC(sampleTaken, sampleToKeep, pop);
                        fit = new FixedIterationTrainer(mimic, 1000);
                        double iterations = fit.train();
                        if(Math.max(ef.value(mimic.getOptimal()), optimalMimic) > optimalMimic) {
                            mimicIterations[j][i] = iterations;
                            mimicTime[j][i] = System.currentTimeMillis() - starttime;
                        }
                        optimalMimic = Math.max(optimalMimic, ef.value(mimic.getOptimal()));
                    }
                }
                mimicRuns[j][i] = optimalMimic;
                System.out.println("MIMIC: " + optimalMimic);
            }
        }

        System.out.println();
        System.out.println(Arrays.deepToString(rhcRuns));
        System.out.println(Arrays.deepToString(saRuns));
        System.out.println(Arrays.deepToString(gaRuns));
        System.out.println(Arrays.deepToString(mimicRuns));

        System.out.println();
        System.out.println(Arrays.deepToString(rhcTime));
        System.out.println(Arrays.deepToString(saTime));
        System.out.println(Arrays.deepToString(gaTime));
        System.out.println(Arrays.deepToString(mimicTime));

        System.out.println();
        System.out.println(Arrays.deepToString(rhcIterations));
        System.out.println(Arrays.deepToString(saIterations));
        System.out.println(Arrays.deepToString(gaIterations));
        System.out.println(Arrays.deepToString(mimicIterations));
    }
}
