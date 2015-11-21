#include <stdio.h>
#include <time.h>
#include <math.h>
#include <omp.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define V 8
#define E 12
#define MAX_WEIGHT 1000000
#define TRUE    1
#define FALSE   0

// __constant__ int M[E];

typedef int boolean;
//
//Represents an edge or path between Vertices
typedef struct Edge
{
    int u;
    int v;

} Edge;

//Represents a Vertex
typedef struct 
{
    int title;
    boolean visited;    

} Vertex;

// __constant__ struct Edge cEdges[E];
    

//Finds the weight of the path from vertex u to vertex v
__device__ __host__ int findEdge(Vertex u, Vertex v, Edge *edges, int *weights)
{

    int i;
    for(i = 0; i < E; i++)
    {
        if(edges[i].u == u.title && edges[i].v == v.title)
        {
            return weights[i];
        }
    }
    return MAX_WEIGHT;
}

int findEdge(Edge *edges, int pu, int pv,  int Size)
{
    int i;
    for(i = 0; i < Size; i++)
    {
        if(edges[i].u == pu && edges[i].v == pv)
        {
            return 1;
        }
    }
    return 0;
}

//Finds the branches of the vertex
__global__ void FindVertexConstant(Vertex *vertices, int *length, int *updateLength)
{

    int u = threadIdx.x;
    if(vertices[u].visited == FALSE)
    {
        vertices[u].visited = TRUE;
        int v;

        for(v = 0; v < V; v++)
        {   
            //Find the weight of the edge
            int weight = findEdge(vertices[u], vertices[v], cEdges, M);

            //Checks if the weight is a candidate
            if(weight < MAX_WEIGHT)
            {   
                //If the weight is shorter than the current weight, replace it
                if(updateLength[v] > length[u] + weight)
                {
                    updateLength[v] = length[u] + weight;
                }
            }
        }
    }
}

__global__ void Find_Vertex(Vertex *vertices, Edge *edges, int *length, int *updateLength)
{
    int u = threadIdx.x;
    if(vertices[u].visited == FALSE)
    {
        vertices[u].visited = TRUE;
        int v;

        for(v = 0; v < V; v++)
        {   
            //Find the weight of the edge
            int weight = findEdge(vertices[u], vertices[v], edges, M);

            //Checks if the weight is a candidate
            if(weight < MAX_WEIGHT)
            {   
                //If the weight is shorter than the current weight, replace it
                if(updateLength[v] > length[u] + weight)
                {
                    updateLength[v] = length[u] + weight;
                }
            }
        }
    }
}

//Updates the shortest path array (length)
__global__ void Update_Paths(Vertex *vertices, int *length, int *updateLength)
{
    int u = threadIdx.x;
    if(length[u] > updateLength[u])
    {

        length[u] = updateLength[u];
        vertices[u].visited = FALSE;
    }

    updateLength[u] = length[u];
}
//Prints the an array of elements

void printArray(int *array)
{
    int i;
    for(i = 0; i < V; i++)
    {
        printf("Shortest Path to Vertex: %d is %d\n", i, array[i]);
    }
}

//Runs the program
int main(void)
{
    int SIZES[] = {10, 20};
    //Variables for the Host Device
    for (int iSize = 0; iSize < sizeof(SIZES)/sizeof(SIZES[0]); ++iSize)
    {
        Vertex *vertices;   
        Edge *edges;

        //Weight of the paths
        int *weights;

        //Len is the shortest path and updateLength is a special array for modifying updates to the shortest path
        int *len, *updateLength;
        
        //Pointers for the CUDA device
        Vertex *d_V;
        Edge *d_E;
        int *d_W;
        int *d_L;
        int *d_C;
      
        int sizeM = sizeof(int)*SIZES[iSize]; 

        //Sizes used for allocation
        // int sizeV = sizeof(Vertex) * V;
        int sizeV = sizeof(Vertex) * (SIZES[iSize] + 1);
        int sizeE = sizeof(Edge) * SIZES[iSize];
        int size = V * sizeof(int);
        //Timer initialization
        float runningTime;
        cudaEvent_t timeStart, timeEnd;
        //Creates the timers
        cudaEventCreate(&timeStart);
        cudaEventCreate(&timeEnd);
        //Allocates space for the variables
        vertices = (Vertex *)malloc(sizeV);
        edges = (Edge *)malloc(sizeE);
        weights = (int *)malloc(SIZES[iSize]* sizeof(int));
        len = (int *)malloc(size);
        updateLength = (int *)malloc(size);

        
        //----------------------------------Graph Base Test-------------------------------------//
        Edge ed[SIZES[iSize]] = {{0, 4}, {0, 6}, {0,2}, {4,6}, {4,7}, {0, 7}, {7, 3}, {3, 1}, {2,5}, {2, 1}, {5,3}, {1, 5}};
        int w[SIZES[iSize]] = {10, 90, 30, 20, 20, 50, 10, 20, 10, 10, 10, 12};

        int i = 0;
        srand(time(NULL));
        for (int i = 0; i < SIZES[iSize]; ++i)
        {
            if 
            ed[i] = {i, i+1};
        }

        for (int ingo = 0; ingo < ceil(SIZES[iSize]/2); ingo++ ){
            int u = rand()%SIZES[iSize];
            int v = rand()%SIZES[iSize];
            if (/* condition */)
            {
                /* code */
            }
            ed[ingo] = {u, v};
        }

        for(i = 0; i < SIZES[iSize]+1; i++)
        {
            Vertex a = { .title =i , .visited=FALSE};
            vertices[i] = a;
        }

        for(i = 0; i < SIZES[iSize]; i++)
        {
            edges[i] = ed[i];
            weights[i] = w[i];
        }

        //----------------------------------Graph Base Test-------------------------------------//
        //--------------------------------Graph Randomizer-----------------------------------//
        // srand(time(NULL));
        // int i = 0;
        // for(i = 0; i < V; i++)
        // {
        //  Vertex a = { .title =(int) i, .visited=FALSE};
        //  vertices[i] = a;
        // }

        // for(i = 0; i < SIZES[iSize]; i++)
        // {

        //  Edge SIZES[iSize] = {.u = (int) rand()%V , .v = rand()%V};
        //  edges[i] = SIZES[iSize];

        //  weights[i] = rand()%100;

        // }

        //--------------------------------Graph Randomizer-----------------------------------//
        //Allocate space on the device
        cudaMalloc((void**)&d_V, sizeV);
        cudaMalloc((void**)&d_E, sizeE);
        cudaMalloc((void**)&d_W, SIZES[iSize] * sizeof(int));
        cudaMalloc((void**)&d_L, size);
        cudaMalloc((void**)&d_C, size);

        //Initial Node
        Vertex root = {0, FALSE};
        //--------------------------------------Dijkstra's Algorithm--------------------------------------//
        root.visited = TRUE;
        
        
        len[root.title] = 0;
        updateLength[root.title] = 0;

        //Copy variables to the Device
        cudaMemcpy(d_V, vertices, sizeV, cudaMemcpyHostToDevice);
        cudaMemcpy(d_E, edges, sizeE, cudaMemcpyHostToDevice);
        cudaMemcpy(d_W, weights, SIZES[iSize] * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_L, len, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_C, updateLength, size, cudaMemcpyHostToDevice);

        cudaMemcpyToSymbol(M,w,sizeM);
        cudaMemcpyToSymbol(cEdges, edges, sizeM);

        

        //Loop finds the initial paths from the node 's'
        for(i = 0; i < V;i++)
        {
            if(vertices[i].title != root.title)
            {
                len[(int)vertices[i].title] = findEdge(root, vertices[i], edges, weights);
                updateLength[vertices[i].title] = len[(int)vertices[i].title];
            }
            else{
                vertices[i].visited = TRUE;
            }
        }

        //Start the timer
        cudaEventRecord(timeStart, 0);
            
        //Recopy the variables  
        cudaMemcpy(d_L, len, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_C, updateLength, size, cudaMemcpyHostToDevice);
                        
        int j;
        //Parallelization
        for(i = 0; i < V; i++){
                //Find_Vertex<<<1, V>>>(d_V, d_E, d_W, d_L, d_C);
            Find_Vertex<<<1, V>>>(d_V, d_E, d_L, d_C);
            // FindVertexConstant<<<1, V>>>(d_V, d_L, d_C);

                for(j = 0; j < V; j++)
                {
                    Update_Paths<<<1,V>>>(d_V, d_L, d_C);
                }
        }   
        
        //Timing Events
        cudaEventRecord(timeEnd, 0);
        cudaEventSynchronize(timeEnd);
        cudaEventElapsedTime(&runningTime, timeStart, timeEnd);

        //Copies the results back
        cudaMemcpy(len, d_L, size, cudaMemcpyDeviceToHost);


        printArray(len);

        //Running Time
        printf("Running Time: %f ms\n", runningTime);
    }

    //--------------------------------------Dijkstra's Algorithm--------------------------------------//

    //Free up the space
    free(vertices);
    free(edges);
    free(weights);
    free(len);
    free(updateLength);
    cudaFree(d_V);
    cudaFree(d_E);
    cudaFree(d_W);
    cudaFree(d_L);
    cudaFree(d_C);
    cudaEventDestroy(timeStart);
    cudaEventDestroy(timeEnd);
}
