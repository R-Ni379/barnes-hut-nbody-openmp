#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <sys/time.h>



//helper struct for returning two function values
struct forces{
    double force_x;
    double force_y;
};

//struct for node
struct node_qt{
    int number_of_particles;
    int particle_placed;
    //index first child
    int index_first_child;
    //index for actually changing particles itself in SoA structure
    int node_particle_index_in_vector;
    //node mass
    double total_mass;
    //center of mass
    double center_of_mass_x;
    double center_of_mass_y;         
    //box measurements
    double box_width;  
    //boxes
    double xmin, ymin; 
    
};


//Velocity Verlet Integration: 2 parts

//initial 
void velocity_verlet_integration_initial(struct node_qt *node_buffer, int current_index, double * restrict position_x, double * restrict position_y,
    double * restrict force_x, double * restrict force_y, double * restrict mass, double * restrict acceleration_x, double * restrict acceleration_y, double * restrict acceleration_x_buffer, 
    double * restrict acceleration_y_buffer, int *quadrant,
    const int N, const double inv_N); 


//function to initialize nodes
void init_Node(struct node_qt *node_buffer, int current_index, double xmin, double ymin, double parent_node_box_width);

//function to build quadtree
void build_quadtree(double *position_x, double *position_y, double *mass, int *quadrant, int N, struct node_qt *node_buffer, int current_index);

//function to insert particles in quadtree
void insert_particles(int index_in_vector, double *position_x, double *position_y, double *mass,int *quadrant, struct node_qt *node_buffer, int current_index);

//function to determine quadrant
static inline int get_quadrant(double position_x, double position_y, struct node_qt *current_node);

//function to split the quadtree
void split_quadtree(struct node_qt *node_buffer, int current_index, double parent_node_box_width);

//function for both, center of mass and mass
void center_of_mass_and_mass(struct node_qt *node_buffer,  double *position_x, double *position_y, double *mass, int current_index);

//force calculation with return value
struct forces calculate_force_qt_ret(double *position_x, double *position_y, double *force_x, double *force_y, double *mass,
                                     struct node_qt *node_buffer, int current_index, int index_in_vector, const int N, const double inv_N, const double theta);

//time loop
void velocity_verlet_integration_timeloop(struct node_qt *node_buffer, int current_index, double * restrict position_x, double * restrict position_y, double * restrict velocity_x, double * restrict velocity_y,
    double * restrict force_x, double * restrict force_y, double * restrict mass, double * restrict acceleration_x, double * restrict acceleration_y, double * restrict acceleration_x_buffer, 
    double * restrict acceleration_y_buffer, const double delta_t, int *quadrant,
    const int N, const double inv_N);


// void function to check wether a particle is out of the bounding box
static inline void check_out_of_bounds_no_ret(double *position_x, double *position_y, int N);


//Time Integration: Symplectic Euler
void calculate_next_state(double * restrict position_x, double * restrict position_y, double * restrict velocity_x, double * restrict velocity_y,
    double * restrict force_x, double * restrict force_y, double * restrict mass, double * restrict acceleration_x, double * restrict acceleration_y,
    int N, double delta_t);



//Symplectic Euler Complete Function
void symplectic_euler_integration(struct node_qt *node_buffer, int current_index, double * restrict position_x, double * restrict position_y, double * restrict velocity_x, double * restrict velocity_y,
    double * restrict force_x, double * restrict force_y, double * restrict mass, double * restrict acceleration_x, double * restrict acceleration_y, 
    const double delta_t, int *quadrant, 
    const int N, const double inv_N);



//Global variable
int index_nxt_avlbl_spot_nd_bffr = 1;



//----------------------------------------------------------------------------------------------

int main(int argc, char *argv[]) {
    // Check if the correct number of command-line arguments are provided
    if (argc != 7) {
        printf("Give 6 input args: N filename nsteps delta_t graphic num_threads\n");
        return -1;
    } 



    // Storing command-line argurments
    const int N = atoi(argv[1]); // Number of particles
    char *filename = argv[2]; // Name of the input file
    const int nsteps = atoi(argv[3]); // Number of time steps
    const double delta_t = atof(argv[4]); // Time step
    int graphics = atoi(argv[5]); // Flag to enable graphics
    int num_threads = atoi(argv[6]); //number of threads
    //make sure to work as serical code if OpenMP not supported
    #ifdef _OPENMP
        //set number of threads 
        omp_set_num_threads(num_threads);
    #else
        num_threads = 0;
    #endif
    
    #ifdef _OPENMP
    printf("%d\n", omp_get_max_threads());
    #else
    printf("Not possible to run OpenMP. Execute program as serial code.");
    #endif


    //SCO: moving them in the order they are used
    double *acceleration_x = malloc(sizeof(double)*N); 
    double *acceleration_y = malloc(sizeof(double)*N); 
    double *force_x = malloc(sizeof(double)*N);
    double *force_y = malloc(sizeof(double)*N);
    double *mass = malloc(sizeof(double)*N);
    double *acceleration_x_buffer = malloc(sizeof(double)*N);
    double *acceleration_y_buffer = malloc(sizeof(double)*N);
    double *position_x = malloc(sizeof(double)*N); 
    double *position_y = malloc(sizeof(double)*N);
    double *velocity_x = malloc(sizeof(double)*N);
    double *velocity_y = malloc(sizeof(double)*N);
    double *brightness= malloc(sizeof(double)*N);
 


    //help to verify in which quadrant the particle landed on which depth 
    int *depth = malloc(sizeof(int)*N);
    int *quadrant = malloc(sizeof(int)*N);
 


    FILE *input_file = fopen(filename, "rb");
    if (input_file == NULL) {
        printf("Error opening file: %s\n", filename);
        return -1;
    }



 
    for (int i = 0; i < N; i++) {
        double data[6];
        if (fread(data, sizeof(double), 6, input_file) != 6) {
            printf("Error reading particle data\n");
            fclose(input_file);
            return -1;
        }

        position_x[i] = data[0];
        position_y[i] = data[1]; 
        mass[i] = data[2];
        velocity_x[i] = data[3];
        velocity_y[i] = data[4];
        brightness[i] = data[5];

    }

    fclose(input_file);

    //NUMA: parallel initialization
    #pragma omp parallel proc_bind(close) num_threads(num_threads) //divide threads onto 8 physical cores on two sockets: run afap -> use spread
    {
        #pragma omp for 
        for (int i = 0; i < N; i++){
            // Initialize computed fields to zero 
            force_x[i] = 0;
            force_y[i] = 0;
            acceleration_x[i] = 0;
            acceleration_y[i] = 0;
            acceleration_x_buffer[i] = 0;
            acceleration_y_buffer[i] = 0;

            depth[i] = 0;
            quadrant[i] = 0;
        }

    }



    //for serial code optimization
    const double inv_N = 1.0/N;

    //allocate big node buffer: calloc is used to initizalie
    struct node_qt *node_buffer = (struct node_qt *) calloc(12*N, sizeof(struct node_qt)); 
    




    //-----SIMULATION---------------------------------------------------------------------------------------------

    printf("VELOCITY VERLET\n");


    //measures all clock time over more cores
    #ifdef _OPENMP
    double start, end;
    start = omp_get_wtime();
    #else
    clock_t start, end;
    start = clock();
   #endif
    //initial call barnes hut and verlet integrator
    velocity_verlet_integration_initial(node_buffer, 0, position_x, position_y, force_x, force_y,
                                        mass, acceleration_x, acceleration_y, acceleration_x_buffer, acceleration_y_buffer,
                                        quadrant, N, inv_N);
    
    
    //create parallel region
    #pragma omp parallel proc_bind(close) num_threads(num_threads) \
    shared(node_buffer, position_x, position_y, velocity_x, velocity_y, \
           acceleration_x, acceleration_y, acceleration_x_buffer, acceleration_y_buffer, \
           quadrant, force_x, force_y, mass, delta_t, N, inv_N)
  
    {
        for (int i = 0; i < nsteps; i++) {
            //start process of barnes hut in time loop using verlet integrator
            velocity_verlet_integration_timeloop(node_buffer, 0, position_x, position_y, velocity_x, velocity_y, force_x, force_y,
                mass, acceleration_x, acceleration_y, acceleration_x_buffer, acceleration_y_buffer,
                delta_t, quadrant, N, inv_N);
            //check if particles are out of bounds
            #pragma omp single
            check_out_of_bounds_no_ret(position_x, position_y, N);
        }
    }

    #ifdef _OPENMP
    end = omp_get_wtime() - start;
    printf("Time taken: %lf seconds\n", end);
    #else
    end = clock();
    double cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC; // Convert to seconds
    printf("Time taken: %f seconds\n", cpu_time_used); 
    #endif


    
    //-------END SIMULATION--------------------------------------------------------------------


    char *output_filename = "result.gal";
    FILE *output_file = fopen(output_filename, "wb");
    if (output_file == NULL) {
        printf("Error opening output file: %s\n", output_filename);
        return -1;
    }

    double data[6];
    for (int i = 0; i < N; i++) {
        data[0] = position_x[i];
        data[1] = position_y[i]; 
        data[2] = mass[i];
        data[3] = velocity_x[i];
        data[4] = velocity_y[i];
        data[5] = brightness[i]; // Adjust this based on the first 6 fields in your struct

        size_t bytes_written = fwrite(data, sizeof(double), 6, output_file);
        if (bytes_written != 6) {
            printf("Error writing particle %d\n", i);
            fclose(output_file);
            return -1;
        }
    }

    fclose(output_file);
    free(position_x);
    free(position_y);
    free(velocity_x);
    free(velocity_y);
    free(force_x);
    free(force_y);
    free(acceleration_x);
    free(acceleration_y);
    free(acceleration_x_buffer);
    free(acceleration_y_buffer);
    free(mass);
    free(brightness);

    free(depth);
    free(quadrant);
    free(node_buffer);
    
    return 0;
}




//VELOCITY VERLET COMPONENTS
void velocity_verlet_integration_initial(struct node_qt *node_buffer, int current_index, double * restrict position_x, double * restrict position_y,
    double * restrict force_x, double * restrict force_y, double * restrict mass, double * restrict acceleration_x, double * restrict acceleration_y, double * restrict acceleration_x_buffer, 
    double * restrict acceleration_y_buffer, int *quadrant,
    const int N, const double inv_N){
        
        //reset buffer
        index_nxt_avlbl_spot_nd_bffr = 1;
        //calculate F(0)
        init_Node(node_buffer, current_index, 0.0, 0.0, 2.0);
        build_quadtree(position_x, position_y, mass, quadrant,  N, node_buffer, current_index);

        struct forces place_holder_force = {0,0}; 
        #pragma omp parallel for schedule(dynamic, 16) private(place_holder_force)
        for (int i = 0; i < N; i++) {
            place_holder_force = calculate_force_qt_ret(position_x, position_y, force_x, force_y, mass, node_buffer, current_index, i, N, inv_N, 0.43);
            force_x[i] = place_holder_force.force_x;  
            force_y[i] = place_holder_force.force_y;
        }


        
        for (int i = 0; i < N; i++) {
            // //a(t)
            acceleration_x[i] = force_x[i]/mass[i];
            acceleration_y[i] = force_y[i]/mass[i];
            // //load a(t) into buffer
            acceleration_x_buffer[i] = acceleration_x[i];
            acceleration_y_buffer[i] = acceleration_y[i];
        }

}



//function to initialize nodes: recuced to 64bytes in order to fit in cache line
void init_Node(struct node_qt *node_buffer, int current_index, double xmin,  double ymin, double parent_node_box_width){
    
    struct node_qt *current_node = &node_buffer[current_index];
    //xmin and ymin of bounding box
    current_node->xmin = xmin; 
    current_node->ymin = ymin;
    //set number of particles in this node
    current_node->number_of_particles = 0;
    //set the index of the first child to -1, i.e. no child yet
    current_node->index_first_child = -1; 
    //particle placed, i.e. no particle placed yet
    current_node->particle_placed= 0;
    //set index in the vectors (position, velocity,...) of the particle lying in this node, i.e. no index yet
    current_node->node_particle_index_in_vector = -1;
    //calculate box width
    current_node->box_width = parent_node_box_width*0.5; 
    //set total mass of the node to 0
    current_node->total_mass = 0.0;
    //set center of mass to 0
    current_node->center_of_mass_x = current_node->center_of_mass_y = 0.0;
}



//build quadtree
void build_quadtree(double *position_x, double *position_y, double *mass, int *quadrant, int N, struct node_qt *node_buffer, int current_index){
    //Insert particles
    for (int i = 0; i < N; i++){
        insert_particles(i, position_x, position_y, mass, quadrant, node_buffer, current_index);
    }
    //calculate mass and center of mass of nodes
    center_of_mass_and_mass(node_buffer, position_x, position_y, mass, current_index); 
}



//insert particles and update center of mass
void insert_particles(int index_in_vector, double *position_x, double *position_y, double *mass, int *quadrant, struct node_qt *node_buffer, int current_index){

    struct node_qt *current_node = &node_buffer[current_index];
    int index_next_child = 0;

    int quad_i = 0; 
    int quad_n = 0;

    // particle end up at same point
    if (current_node->particle_placed == 1 && position_x[current_node->node_particle_index_in_vector] == position_x[index_in_vector] && position_y[current_node->node_particle_index_in_vector] == position_y[index_in_vector]){
        printf("ERROR: Particles ended up at the same point. Free memory and abort Simulation.\n");
        exit(1);
    }


    //MORE THAN 1 PARTICLE -> I.E. Node has already been created
    else if (current_node->number_of_particles > 1){ 
        //printf("More Particles\n");
        quad_i = get_quadrant(position_x[index_in_vector], position_y[index_in_vector], current_node); 
        //increase number of particle for this node 
        current_node->number_of_particles += 1;

        //new particle
        //Quadrant I
        if (quad_i == 1){ 
            quadrant[index_in_vector] = 1;
            index_next_child = current_node->index_first_child;
            insert_particles(index_in_vector, position_x, position_y, mass, quadrant, node_buffer, index_next_child); 
        }
        //Quadrant II
        if (quad_i == 2){ 
            quadrant[index_in_vector] = 2;
            index_next_child = current_node->index_first_child+1;
            insert_particles(index_in_vector, position_x, position_y, mass, quadrant, node_buffer, index_next_child);
        }
        //Quadrant III
        if (quad_i == 3){ 
            quadrant[index_in_vector] = 3;
            index_next_child = current_node->index_first_child+2;
            insert_particles(index_in_vector, position_x, position_y, mass, quadrant, node_buffer, index_next_child);
        }
        //Quadrant IV
        if (quad_i == 4){ 
            quadrant[index_in_vector] = 4;
            index_next_child = current_node->index_first_child+3;
            insert_particles(index_in_vector, position_x, position_y, mass, quadrant, node_buffer, index_next_child);
        }
    } 


    //1 PARTICLE
    else if (current_node->number_of_particles == 1) { 
        quad_n = get_quadrant(position_x[current_node->node_particle_index_in_vector], position_y[current_node->node_particle_index_in_vector], current_node);
        quad_i = get_quadrant(position_x[index_in_vector], position_y[index_in_vector], current_node);
        //split and becomes parents
        current_node->index_first_child = index_nxt_avlbl_spot_nd_bffr; 
        //split quadtree
        split_quadtree(node_buffer, current_index, current_node->box_width);
        //existing particle
        //Quadrant I
        if (quad_n == 1){
            quadrant[current_node->node_particle_index_in_vector] = 1;
            index_next_child = current_node->index_first_child; 
            insert_particles(current_node->node_particle_index_in_vector, position_x, position_y, mass , quadrant, 
                            node_buffer, index_next_child); 
        }
        //Quadrant II
        if (quad_n == 2){
            quadrant[current_node->node_particle_index_in_vector] = 2;
            index_next_child = current_node->index_first_child+1;
            insert_particles(current_node->node_particle_index_in_vector, position_x, position_y, mass , quadrant, 
                            node_buffer, index_next_child);
        }
        //Quadrant III
        if (quad_n == 3){
            quadrant[current_node->node_particle_index_in_vector] = 3;
            index_next_child = current_node->index_first_child+2;
            insert_particles(current_node->node_particle_index_in_vector, position_x, position_y, mass , quadrant, 
                            node_buffer, index_next_child);
        }
        //Quadrant IV
        if (quad_n == 4){ 
            quadrant[current_node->node_particle_index_in_vector] = 4;
            index_next_child = current_node->index_first_child+3;
            insert_particles(current_node->node_particle_index_in_vector, position_x, position_y, mass , quadrant, 
                            node_buffer, index_next_child);
        }
        //"remove" particles that is currently in this node
        current_node->particle_placed = 0;

        //new particle
        //increase number of particle for this node
        current_node->number_of_particles += 1;
        //Quadrant I 
        if (quad_i == 1){
            quadrant[index_in_vector] = 1;
            index_next_child = current_node->index_first_child;
            insert_particles(index_in_vector, position_x, position_y, mass, quadrant, node_buffer, index_next_child);
        }
        //Quadrant II
        if (quad_i == 2){
            quadrant[index_in_vector] = 2;
            index_next_child = current_node->index_first_child+1;
            insert_particles(index_in_vector, position_x, position_y, mass, quadrant, node_buffer, index_next_child);
        }
        //Quadrant III
        if (quad_i == 3){
            quadrant[index_in_vector] = 3;
            index_next_child = current_node->index_first_child+2;
            insert_particles(index_in_vector, position_x, position_y, mass, quadrant, node_buffer, index_next_child);
        }
        //Quadrant IV  
        if (quad_i == 4){
            quadrant[index_in_vector] = 4;
            index_next_child = current_node->index_first_child+3;
            insert_particles(index_in_vector, position_x, position_y, mass, quadrant, node_buffer, index_next_child);
        }

        return;

    }

    //EMPTY node -> Leaf node
    else if (current_node->number_of_particles == 0){  
        current_node->particle_placed = 1;

        //index based idea: need to have access to original source
        current_node->node_particle_index_in_vector = index_in_vector;
        mass[current_node->node_particle_index_in_vector] = mass[index_in_vector];
        quadrant[current_node->node_particle_index_in_vector] = quadrant[index_in_vector];
        current_node->number_of_particles += 1;
       
        //set first child to -1 
        current_node->index_first_child = -1;  
        return;
    }

}


//determine quadrant 
static inline int get_quadrant(double position_x, double position_y, struct node_qt *current_node){
    int quadrant = 0;
        //Q1 
        if (position_x <= current_node->xmin + current_node->box_width*0.5 && position_y <= current_node->ymin + current_node->box_width*0.5)   
        quadrant = 1;
        //Q2
        else if (position_x > current_node->xmin + current_node->box_width*0.5 && position_y <= current_node->ymin + current_node->box_width*0.5)
            quadrant = 2;
        //Q3
        else if (position_x > current_node->xmin + current_node->box_width*0.5 && position_y > current_node->ymin + current_node->box_width*0.5)
            quadrant = 3;
        //Q4
        else
            quadrant = 4;
        
        
        return quadrant;
} 

//split quadtree 
void split_quadtree(struct node_qt *node_buffer, int current_index, double parent_node_box_width){
    struct node_qt *current_node = &node_buffer[current_index];
    //Q1 / SW / ll
    init_Node(node_buffer, index_nxt_avlbl_spot_nd_bffr, current_node->xmin, current_node->ymin, parent_node_box_width);
    //Q2 / SE / rl
    init_Node(node_buffer, index_nxt_avlbl_spot_nd_bffr+1, current_node->xmin + current_node->box_width*0.5,  current_node->ymin, parent_node_box_width);
    //Q3 / NE / ru
    init_Node(node_buffer, index_nxt_avlbl_spot_nd_bffr+2, current_node->xmin + current_node->box_width*0.5, current_node->ymin + current_node->box_width*0.5, parent_node_box_width);
    //Q4 / NW / lu
    init_Node(node_buffer, index_nxt_avlbl_spot_nd_bffr+3, current_node->xmin, current_node->ymin + current_node->box_width*0.5,  parent_node_box_width);
    
    //bump counter forward 
    index_nxt_avlbl_spot_nd_bffr += 4;  
    
}

//function for both, mass and center of mass 
void center_of_mass_and_mass(struct node_qt *node_buffer,  double *position_x, double *position_y, double *mass, int current_index){
    //get current node (defined by the whole buffer and the current index)
    struct node_qt *current_node = &node_buffer[current_index];
    //leaf node containing 1 particle
    if (current_node->number_of_particles == 1){
        current_node->center_of_mass_x = position_x[current_node->node_particle_index_in_vector];
        current_node->center_of_mass_y = position_y[current_node->node_particle_index_in_vector];
        current_node->total_mass = mass[current_node->node_particle_index_in_vector]; 
    }
    //leaf node containing 0 particles -> set to 0
    else if (current_node->number_of_particles == 0){
        current_node->center_of_mass_x = 0;
        current_node->center_of_mass_y = 0;
        current_node->total_mass = 0; 
    }
    else{
        //recursive call to left lower child
        center_of_mass_and_mass(node_buffer, position_x, position_y, mass, current_node->index_first_child);  
        //recursive call to right lower child
        center_of_mass_and_mass(node_buffer, position_x, position_y, mass, current_node->index_first_child+1);
        //recursive call to right upper child
        center_of_mass_and_mass(node_buffer, position_x, position_y, mass, current_node->index_first_child+2);
        //recursive call to left upper child
        center_of_mass_and_mass(node_buffer, position_x, position_y, mass, current_node->index_first_child+3);

        //local variables for calculations
        struct node_qt *ll = &node_buffer[current_node->index_first_child];  
        struct node_qt *rl = &node_buffer[current_node->index_first_child+1];
        struct node_qt *ru = &node_buffer[current_node->index_first_child+2];
        struct node_qt *lu = &node_buffer[current_node->index_first_child+3];

        //calculate total mass and center of mass of the current node
        current_node->total_mass = ll->total_mass + rl->total_mass + ru->total_mass + lu->total_mass;
        current_node->center_of_mass_x = ((ll->total_mass * ll->center_of_mass_x) + (rl->total_mass * rl->center_of_mass_x) + (ru->total_mass * ru->center_of_mass_x) + (lu->total_mass * lu->center_of_mass_x))/current_node->total_mass;
        current_node->center_of_mass_y = ((ll->total_mass * ll->center_of_mass_y) + (rl->total_mass * rl->center_of_mass_y) + (ru->total_mass * ru->center_of_mass_y) + (lu->total_mass * lu->center_of_mass_y))/current_node->total_mass;


    }   
        
}


//force calculation
struct forces calculate_force_qt_ret(double *position_x, double *position_y, double *force_x, double *force_y, double *mass,
                                    struct node_qt *node_buffer, int current_index, int index_in_vector, const int N, const double inv_N, const double theta){

    //get current node (defined by the whole buffer and the current index)
    struct node_qt *current_node = &node_buffer[current_index];
    const double G = -100.0 * inv_N;  
    const double e_0 = 0.001; 
    //local variables for calculations
    struct forces aux_force = {0,0};
    struct forces aux_force_ll = {0, 0};
    struct forces aux_force_rl = {0, 0};
    struct forces aux_force_ru = {0, 0};
    struct forces aux_force_lu = {0, 0};
    

    
    //leaf node containing the particle to be inserted itself                                
    if (current_node->particle_placed == 1 && position_x[current_node->node_particle_index_in_vector] == position_x[index_in_vector] && position_y[current_node->node_particle_index_in_vector] == position_y[index_in_vector]){      
        aux_force.force_x = 0;
        aux_force.force_y = 0;
    }

    //leaf node 
    else if(current_node->index_first_child == -1){ 
        //calculate vectors
        const double x_vector = position_x[index_in_vector] - current_node->center_of_mass_x;
        const double y_vector = position_y[index_in_vector] - current_node->center_of_mass_y;
        const double r = sqrt(x_vector*x_vector + y_vector*y_vector);
        const double r_epsilon_cubed = (r + e_0) * (r + e_0) * (r + e_0);
        const double r_inv_epsilon_cubed = 1.0/r_epsilon_cubed;

        //calculate forces
        aux_force.force_x = G * mass[index_in_vector] * current_node->total_mass * x_vector * r_inv_epsilon_cubed;
        aux_force.force_y = G * mass[index_in_vector] * current_node->total_mass * y_vector * r_inv_epsilon_cubed;
        
    }
        
    //no leaf node
    else{
        //calculate vectors
        const double x_vector = position_x[index_in_vector] - current_node->center_of_mass_x;
        const double y_vector = position_y[index_in_vector] - current_node->center_of_mass_y;
        const double r = sqrt(x_vector*x_vector + y_vector*y_vector);
        //needed for condition check
        const double r_inv = 1.0/r;

        //group of particles sufficiently far away to be grouped together, should be called theta_max instead of theta to avoid confusion
        if (current_node->box_width * r_inv <= theta) {
            const double r_epsilon_cubed = (r + e_0) * (r + e_0) * (r + e_0);
            const double r_inv_epsilon_cubed = 1.0/r_epsilon_cubed;
            //calculate forces
            aux_force.force_x = G * mass[index_in_vector] * current_node->total_mass * x_vector * r_inv_epsilon_cubed;
            aux_force.force_y = G * mass[index_in_vector] * current_node->total_mass * y_vector * r_inv_epsilon_cubed;
        
        } 
        //group of particles not sufficiently far away -> traverse down to the quadtree
        else {
            //recursive call to left lower child
            aux_force_ll = calculate_force_qt_ret(position_x, position_y, force_x, force_y, mass, node_buffer, current_node->index_first_child, index_in_vector, N, inv_N, theta);
            //recursive call to right lower child
            aux_force_rl = calculate_force_qt_ret(position_x, position_y, force_x, force_y, mass, node_buffer, current_node->index_first_child+1, index_in_vector, N, inv_N, theta);
            //recursive call to right upper child
            aux_force_ru = calculate_force_qt_ret(position_x, position_y, force_x, force_y, mass, node_buffer, current_node->index_first_child+2, index_in_vector, N, inv_N, theta);
            //recursive call to left upper child
            aux_force_lu = calculate_force_qt_ret(position_x, position_y, force_x, force_y, mass, node_buffer, current_node->index_first_child+3, index_in_vector, N, inv_N, theta);

            //sum up the forces of the 4 child nodes
            aux_force.force_x = aux_force_ll.force_x + aux_force_rl.force_x + aux_force_ru.force_x + aux_force_lu.force_x;
            aux_force.force_y = aux_force_ll.force_y + aux_force_rl.force_y + aux_force_ru.force_y + aux_force_lu.force_y;
            }
        }

        //return forces on particle i in x and y direction 
        return aux_force;
        
}



void velocity_verlet_integration_timeloop(struct node_qt *node_buffer, int current_index, double * restrict position_x, double * restrict position_y, double * restrict velocity_x, double * restrict velocity_y,
    double * restrict force_x, double * restrict force_y, double * restrict mass, double * restrict acceleration_x, double * restrict acceleration_y, double * restrict acceleration_x_buffer, 
    double * restrict acceleration_y_buffer, const double delta_t, int *quadrant,
    const int N, const double inv_N){

        #pragma omp for schedule(static)
        for (int i = 0; i < N; i++){  //VECTORIZED   
            position_x[i] += velocity_x[i] * delta_t + 0.5 * acceleration_x[i] * delta_t * delta_t;
            position_y[i] += velocity_y[i] * delta_t + 0.5 * acceleration_y[i] * delta_t * delta_t;
        } 
    
        #pragma omp single
        {
        //reset global variable
        index_nxt_avlbl_spot_nd_bffr = 1;   

        init_Node(node_buffer, current_index, 0.0, 0.0, 2.0);
        build_quadtree(position_x, position_y, mass, quadrant, N, node_buffer, current_index);
        }
        #pragma omp barrier

        struct forces place_holder_force_f = {0,0}; 
        #pragma omp for schedule(dynamic, 16) private(place_holder_force_f)
        for (int i = 0; i < N; i++){
                    place_holder_force_f = calculate_force_qt_ret(position_x, position_y, force_x, force_y, mass, node_buffer, current_index, i, N, inv_N, 0.43); 
                    force_x[i] = place_holder_force_f.force_x;  
                    force_y[i] = place_holder_force_f.force_y;
        }


        //calculate a(t+delta_t) and v(t+delta_t)
        #pragma omp for schedule(static)
        for (int i = 0; i < N; i++){  //VECTORIZED   
            //calculate a(t+delta_t)
            acceleration_x[i] = force_x[i]/mass[i];
            acceleration_y[i] = force_y[i]/mass[i];
            //calculate v(t+delta_t)
            velocity_x[i] += 0.5 * (acceleration_x_buffer[i] + acceleration_x[i]) * delta_t;
            velocity_y[i] += 0.5 * (acceleration_y_buffer[i] + acceleration_y[i]) * delta_t;
            //load into buffer
            acceleration_x_buffer[i] = acceleration_x[i];  
            acceleration_y_buffer[i] = acceleration_y[i];

        }

}

//check out of bounds
static inline void check_out_of_bounds_no_ret(double *position_x, double *position_y, int N){
    for (int i = 0; i < N; i++){
            if (position_x[i] > 1 || position_x[i]  < 0 || position_y[i] > 1 || position_y[i] < 0){
                printf("ERROR: Particle outside the bounding box. Stop simulation.\n");
                exit(1);
            }
       }
}


