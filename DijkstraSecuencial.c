//  Author: Jose F. Martinez Rivera
//  Course: ICOM4036 - 040
//  Professor: Wilson Rivera Gallego
//  Assignment 2 - OpenMP Implementation

#include <stdio.h>
#include <time.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <cuda.h>
#include <stdlib.h>
#include <sys/time.h>
#include <iostream>
#include <highgui.h>
#include <cv.h>

#define MAXINT 100000
#define TRUE    1
#define FALSE   0
#define V 24
#define E 36
#define USECPSEC 1000000ULL

//boolean type
typedef int bool;

//Represents an edge or path between Vertices
typedef struct
{
    int u;
    int v;

} Edge;

//Represents a Vertex
typedef struct 
{
    int title;
    bool visited;   

} Vertex;


unsigned long long dtime_usec(unsigned long long prev){
  timeval tv1;
  gettimeofday(&tv1,0);
  return ((tv1.tv_sec * USECPSEC)+tv1.tv_usec) - prev;
  // return ((tv1.tv_sec *1000)+tv1.tv_usec/1000) - prev;
}
//Prints the array
void printArray(int *array)
{
    int i;
    for(i = 0; i < V; i++)
    {
        printf("Path to Vertex %d is %d\n", i, array[i]);
    }
}

//OpenMP Implementation of Dijkstra's Algorithm
void DijkstraOMP(Vertex *vertices, Edge *edges, int *weights, Vertex *root)
{   

    double start, end;
    unsigned long long cpu_time = dtime_usec(0);
    root->visited = TRUE;
    
    int len[V];
    len[(int)root->title] = 0;

    int i, j;
    
    for(i = 0; i < V;i++)
    {

        if(vertices[i].title != root->title)
        {
            len[(int)vertices[i].title] = findEdge(*root, vertices[i], edges, weights);
            

        }
        else{
        
            vertices[i].visited = TRUE;
        }

    }

    
    for(j = 0; j < V; j++){

        
        Vertex u;
        int h = minPath(vertices, len);
        u = vertices[h];
        
        
            for(i = 0; i < V; i++)
            {
                if(vertices[i].visited == FALSE)
                {   
                    int c = findEdge( u, vertices[i], edges, weights);
                    len[vertices[i].title] = minimum(len[vertices[i].title], len[u.title] + c);
                
                }
            }
    }
    printArray(len);
    cpu_time = dtime_usec(cpu_time);
    printf("Finished 1. Basic.  Results match. gpu time: %lld\n", cpu_time);
    // printf("Running time: %f ms\n", (end - start)*1000);
    
    
    
}

//Finds the edge that connects Vertex u with Vertex v
int findEdge(Vertex u, Vertex v, Edge *edges, int *weights)
{

    int i;
    for(i = 0; i < E; i++)
    {

        if(edges[i].u == u.title && edges[i].v == v.title)
        {
            return weights[i];
        }
    }

    return MAXINT;

}

//Returns the minimum between two integers
int minimum(int A, int B)
{
    if( A > B)
    {
        return B;
    }

    else{
        return A;
    }
}

//Visits the vertices and looks for the lowest weight from the vertex
int minWeight(int *len, Vertex *vertices)
{
    int i;
    int minimum = MAXINT;
    for(i = 0; i < V; i++)
    {
        if(vertices[i].visited == TRUE)
        {
            continue;
        }
        
        else if(vertices[i].visited == FALSE && len[i] < minimum)
        {
            minimum = len[i];
            
        }
        
    }
    return minimum;
}

//Localizes the vertex with the lowest weight path
int minPath(Vertex *vertices, int *len)
{
    int i;
    int min = minWeight(len, vertices);
    
    for(i = 0; i < V; i++)
    {
        if(vertices[i].visited == FALSE && len[vertices[i].title] == min)
        {
            vertices[i].visited = TRUE;
            return i;

        }
    }
    
    
    
    
}

int main(void)
{

    Vertex vertices[V];
    // Edge edges[E];
    // int weights[E];
    
    
    //----------------------------------Graph Base Test-------------------------------------//

    // Edge edges[E] ={{0, 4}, {0, 6}, {0,2}, {4,6}, {4,7}, {0, 7}, {7, 3}, {3, 1}, {2,5}, {2, 1}, {5,3}};
    // int weights[E] = {10, 90, 30, 20, 20, 50, 10, 20, 10, 10, 10};

    Edge edges[E] = {{0, 4}, {0, 6}, {0,2}, {4,6}, {4,7}, {0, 7}, {7, 3}, {3, 1}, {2,5}, {2, 1}, {5,3}, {2,6}, {6,9}, {5,12}, {8,10}, {12,15}, {10,12}, {7, 9}, {7, 10}, {8, 10}, {7,8 },{8,9 },{9,10 },{10,11 },{11,12 },{12,13 },{13,14 },{14,15 },{15,16 },{16,17 },{17,18 },{18,19},{19,20 },{20,21},{21,22 },{22, 23}};
    int weights[E] = {10, 90, 30, 20, 20, 50, 10, 20, 10, 10, 10, 70, 10, 90, 30, 20, 20, 50, 10, 20, 10, 10, 10, 20, 10, 90, 30, 20, 20, 50, 10, 20, 10, 10, 10, 20};


    int i = 0;
    for(i = 0; i < V; i++)
    {
        Vertex a = { .title =i , .visited=FALSE};
        vertices[i] = a;

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

    Vertex root = {0, FALSE};

    printf("OpenMP Results for Small Graph of 8 Vertices:\n");
    DijkstraOMP(vertices, edges, weights, &root);

    
    

}

    
    
    
    
    
    
    
    
    
    
    
    
    
