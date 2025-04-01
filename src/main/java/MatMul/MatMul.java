package MatMul;

import uk.ac.manchester.tornado.api.annotations.Parallel;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.TornadoExecutionResult;
import uk.ac.manchester.tornado.api.types.matrix.Matrix2DFloat;
import uk.ac.manchester.tornado.api.WorkerGrid2D;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.KernelContext;

public class MatMul {
    private static final int SIZE = 1024;

    // mxm serial
    public static void MultiplySerial(Matrix2DFloat A, Matrix2DFloat B, Matrix2DFloat C)
    {
        for (int i = 0; i < SIZE; i++) {
            for (int j = 0; j < SIZE; j++) {
                float sum = 0.0f;
                for (int k = 0; k < SIZE; k++) {
                    sum += A.get(i,k) * B.get(k,j);
                }
                C.set(i,j,sum);
            }
        }        
    }

    // mxm loop
    public static void MultiplyLoop(Matrix2DFloat A, Matrix2DFloat B, Matrix2DFloat C) 
    {
        for (@Parallel int i = 0; i < SIZE; i++) {
            for (@Parallel int j = 0; j < SIZE; j++) {
                float sum = 0.0f;
                for (int k = 0; k < SIZE; k++) {
                    sum += A.get(i,k) * B.get(k,j);
                }
                C.set(i,j,sum);
            }
        }
    }

    // mxm loop fold
    public static void MultiplyLoopFold(Matrix2DFloat A, Matrix2DFloat B, Matrix2DFloat C) 
    {
        for (@Parallel int iter = 0; iter < SIZE*SIZE; iter++) {
            int i=iter/SIZE;
            int j=iter%SIZE;
            float sum = 0.0f;
            for (int k = 0; k < SIZE; k++) {
                sum += A.get(i,k) * B.get(k,j);
            }
            C.set(i,j,sum);
        }
    }

    // mxm kernel
    public static void MultiplyKernel(KernelContext context, Matrix2DFloat A, Matrix2DFloat B, Matrix2DFloat C)
    {
        int idx=context.localIdx;
        int jdx=context.localIdy;
        float sum=0.0f;
        for(int k=0;k<SIZE;k++)
        {
            sum+=A.get(idx,k)*B.get(k,jdx);
        }
        C.set(idx,jdx,sum);
    }

    public static void main(String[] args) 
    {
        
        // init matrix and filled with random data
        Matrix2DFloat ASerial = new Matrix2DFloat(SIZE,SIZE);
        Matrix2DFloat BSerial = new Matrix2DFloat(SIZE,SIZE);
        Matrix2DFloat CSerial = new Matrix2DFloat(SIZE,SIZE);
        Matrix2DFloat AParallel = new Matrix2DFloat(SIZE,SIZE);
        Matrix2DFloat BParallel = new Matrix2DFloat(SIZE,SIZE);
        Matrix2DFloat CParallel = new Matrix2DFloat(SIZE,SIZE);
        Matrix2DFloat AParallelfold = new Matrix2DFloat(SIZE,SIZE);
        Matrix2DFloat BParallelfold = new Matrix2DFloat(SIZE,SIZE);
        Matrix2DFloat CParallelfold = new Matrix2DFloat(SIZE,SIZE);
        Matrix2DFloat AKernel = new Matrix2DFloat(SIZE,SIZE);
        Matrix2DFloat BKernel = new Matrix2DFloat(SIZE,SIZE);
        Matrix2DFloat CKernel = new Matrix2DFloat(SIZE,SIZE);
        for(int i=0;i<SIZE;i++)
        {
            for(int j=0;j<SIZE;j++)
            {
                ASerial.set(i,j,(float)Math.random());
                BSerial.set(i,j,(float)Math.random());
                AParallel.set(i,j,ASerial.get(i,j));
                BParallel.set(i,j,BSerial.get(i,j));
                AParallelfold.set(i,j,ASerial.get(i,j));
                BParallelfold.set(i,j,BSerial.get(i,j));
                AKernel.set(i,j,ASerial.get(i,j));
                BKernel.set(i,j,BSerial.get(i,j));
            }
        }

        // serial
        long startSerial=System.nanoTime();
        MultiplySerial(ASerial,BSerial,CSerial);
        long endSerial=System.nanoTime();

        // loop parallel
        TaskGraph taskGraphLoopParallel =new TaskGraph("s0")
            .transferToDevice(DataTransferMode.FIRST_EXECUTION,AParallel,BParallel)
            .task("t0",MatMul::MultiplyLoop,AParallel,BParallel,CParallel)
            .transferToHost(DataTransferMode.EVERY_EXECUTION,CParallel);
        ImmutableTaskGraph immutableTaskGraphLoopParallel=taskGraphLoopParallel.snapshot();
        TornadoExecutionPlan executionPlanLoopParallel=new TornadoExecutionPlan(immutableTaskGraphLoopParallel);
        long startLoopParallel=System.nanoTime();
        executionPlanLoopParallel.execute();
        long endLoopParallel=System.nanoTime();

        // loop parallel fold
        TaskGraph taskGraphFoldParallel=new TaskGraph("s1")
            .transferToDevice(DataTransferMode.FIRST_EXECUTION,AParallelfold,BParallelfold)
            .task("t0",MatMul::MultiplyLoopFold,AParallelfold,BParallelfold,CParallelfold)
            .transferToHost(DataTransferMode.EVERY_EXECUTION,CParallelfold);
        ImmutableTaskGraph immutableTaskGraphFoldParallel=taskGraphFoldParallel.snapshot();
        TornadoExecutionPlan executionPlanFoldParallel=new TornadoExecutionPlan(immutableTaskGraphFoldParallel);
        long startFoldParallel=System.nanoTime();
        executionPlanFoldParallel.execute();
        long endFoldParallel=System.nanoTime();
        
        // kernel
        WorkerGrid2D workerGrid = new WorkerGrid2D(SIZE,SIZE);
        GridScheduler gridScheduler =new GridScheduler("s2.t0",workerGrid);
        KernelContext context=new KernelContext();
        workerGrid.setLocalWork(32,32,1);
        TaskGraph taskGraphKernel =new TaskGraph("s2")
            .transferToDevice(DataTransferMode.FIRST_EXECUTION,AKernel,BKernel)
            .task("t0",MatMul::MultiplyKernel,context,AKernel,BKernel,CKernel)
            .transferToHost(DataTransferMode.EVERY_EXECUTION,CKernel);
        ImmutableTaskGraph immutableTaskGraphKernel=taskGraphKernel.snapshot();
        TornadoExecutionPlan executionPlanKernel=new TornadoExecutionPlan(immutableTaskGraphKernel);
        long startKernel=System.nanoTime();
        executionPlanKernel.withGridScheduler(gridScheduler).execute();
        long endKernel=System.nanoTime();

        // correct
        int pcorrect=1;
        int pfcorrect=1;
        int kcorrect=1;
        for(int i=0;i<SIZE;i++)
        {
            for(int j=0;j<SIZE;j++)
            {
                if(CParallel.get(i,j)-CSerial.get(i,j)>1e6)
                {
                    System.out.printf("parallel wrong! %d %d %.10f %.10f\n",i,j,CParallel.get(i,j),CSerial.get(i,j));
                    pcorrect=0;
                }
                if(CParallelfold.get(i,j)-CSerial.get(i,j)>1e6)
                {
                    System.out.printf("kernel wrong! %d %d %.10f %.10f\n",i,j,CKernel.get(i,j),CSerial.get(i,j));
                    pfcorrect=0;
                }
                if(CKernel.get(i,j)-CSerial.get(i,j)>1e6)
                {
                    System.out.printf("kernel wrong! %d %d %.10f %.10f\n",i,j,CKernel.get(i,j),CSerial.get(i,j));
                    kcorrect=0;
                }
                if(pcorrect!=1||pfcorrect!=1||kcorrect!=1)
                {
                    break;
                }
            }
            if(pcorrect!=1||kcorrect!=1)
            {
                break;
            }
        }

        System.out.printf("Serial: %.3f ms\n",(endSerial-startSerial)/1e6);
        System.out.printf("loop: %.3f ms\n",(endLoopParallel-startLoopParallel)/1e6);
        System.out.printf("loop fold: %.3f ms\n",(endFoldParallel-startFoldParallel)/1e6);
        System.out.printf("kernle: %.3f ms\n",(endKernel-startKernel)/1e6);
    }
}