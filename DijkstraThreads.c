#include <stdio.h>
#include <time.h>
#include <math.h>
#include <omp.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define V 24
#define E 36
#define MAX_WEIGHT 1000000
#define TRUE    1
#define FALSE   0

__constant__ int M[E];

typedef int boolean;
//
//Represents an edge or path between Vertices
typedef struct Edge
{
    int u;
    int v;
    // int weight;
} Edge;

//Represents a Vertex
typedef struct 
{
    int title;
    boolean visited;    

} Vertex;

__constant__ struct Edge cEdges[E];
    

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
            //Find the weight of the edge adjacent to u 
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
    // int SIZES[] = {51, 101};
    //Variables for the Host Device
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
  
    int sizeM = sizeof(int)*E; 

    //Sizes used for allocation
    int sizeV = sizeof(Vertex) * V;
    int sizeE = sizeof(Edge) * E;
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
    weights = (int *)malloc(E* sizeof(int));
    len = (int *)malloc(size);
    updateLength = (int *)malloc(size);

    
    //----------------------------------Graph Base Test-------------------------------------//
    Edge ed[E] = {{0, 4}, {0, 6}, {0,2}, {4,6}, {4,7}, {0, 7}, {7, 3}, {3, 1}, {2,5}, {2, 1}, {5,3}, {2,6}, {6,9}, {5,12}, {8,10}, {12,15}, {10,12}, {7, 9}, {7, 10}, {8, 10}, {7,8 },{8,9 },{9,10 },{10,11 },{11,12 },{12,13 },{13,14 },{14,15 },{15,16 },{16,17 },{17,18 },{18,19},{19,20 },{20,21},{21,22 },{22, 23}};
    int w[E] = {10, 90, 30, 20, 20, 50, 10, 20, 10, 10, 10, 70, 10, 90, 30, 20, 20, 50, 10, 20, 10, 10, 10, 20, 10, 90, 30, 20, 20, 50, 10, 20, 10, 10, 10, 20};

    int i = 0;
    for(i = 0; i < V; i++)
    {
        Vertex a = { .title =i , .visited=FALSE};
        vertices[i] = a;
    }

    for(i = 0; i < E; i++)
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

    // for(i = 0; i < E; i++)
    // {

    //  Edge e = {.u = (int) rand()%V , .v = rand()%V};
    //  edges[i] = e;

    //  weights[i] = rand()%100;

    // }

    //--------------------------------Graph Randomizer-----------------------------------//
    //Allocate space on the device
    cudaMalloc((void**)&d_V, sizeV);
    cudaMalloc((void**)&d_E, sizeE);
    cudaMalloc((void**)&d_W, E * sizeof(int));
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
    cudaMemcpy(d_W, weights, E * sizeof(int), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_L, len, size, cudaMemcpyHostToDevice);
    // cudaMemcpy(d_C, updateLength, size, cudaMemcpyHostToDevice);

    cudaMemcpyToSymbol(M,w,sizeM);
    cudaMemcpyToSymbol(cEdges, edges, sizeM);

    

    //Loop finds the initial paths from the node 's'
    for(i = 0; i < V;i++)
    {
        if(vertices[i].title != root.title)
        {
            len[vertices[i].title] = findEdge(root, vertices[i], edges, weights);
            updateLength[vertices[i].title] = len[vertices[i].title];
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

            // for(j = 0; j < V; j++)
            // {
        Update_Paths<<<1,V>>>(d_V, d_L, d_C);
            // }
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
