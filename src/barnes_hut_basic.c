#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>


//struct for particles
typedef struct {
    //particle ID
    int particle_ID;
    // Values from input file
    double position_x; // x-coordinate of a particle
    double position_y; // y-coordinate of a particle
    double mass; // mass of a particle
    double velocity_x; // x-velocity of a particle
    double velocity_y; // y-velocity of a particle
    double brightness; // brightness of a particle

    // Values that are calculated
    double force_x; // x-force acting on a particle
    double force_y; // y-force acting on a particle 
    double acceleration_x; // x-force acting on a particle
    double acceleration_y; // y-force acting on a particle
    double acceleration_x_buffer;
    double acceleration_y_buffer;

    //help to verify in which quadrant the particle landed on which depth
    int depth;
    int quadrant;

} Particle;

//helper struct for returning two function values
struct forces{
    double force_x;
    double force_y;
};

//struct for node
struct node_qt{
    //node_ID; 
    int depth;
    int number_of_particles;
    double total_mass;
    //box measurements
    double box_width;
    //center of mass
    double center_of_mass_x;
    double center_of_mass_y;
    //boxes
    double xmin, xmax, ymin, ymax;
    //particle
    Particle *particle; 
    
    //child nodes
    struct node_qt *lu; //left upper
    struct node_qt *ru; //right upper
    struct node_qt *ll; //left lower
    struct node_qt *rl; //right lower
};




//force calculation with return value
struct forces calculate_force_qt_ret(Particle *particle_i, struct node_qt **node, const int N, const double theta);

//function to create node
struct node_qt *create_Node(double xmin, double xmax, double ymin, double ymax, int depth);

//function to build quadtree
void build_quadtree(Particle *particles, int N, struct node_qt **node);

//function to insert particles in quadtree
void insert_particles(Particle *particle_i, struct node_qt **node);

//function  to determine quadrant  
int get_quadrant(Particle *particle, struct node_qt **node);

//function to split the quadtree
void split_quadtree(struct node_qt **node);

 

//function to check wether a particle is out of the bounding box
int check_out_of_bounds(Particle *particles, int N);

//destroy quadtree -> free all memory
void destroy_quadtree(struct node_qt **node);

//another approach to destroy tree 
void delete_tree(struct node_qt **node);

//print function
void print_data(struct node_qt **node);

//function for both, center of mass and mass
void center_of_mass_and_mass(struct node_qt **node);


//Time Integration: Symplectic Euler
void calculate_next_state(Particle *particles, const int N, const double delta_t);

//Velocity Verlet Time Integration
void velocity_verlet_next_state(Particle *particles, const int N, const double delta_t);

//Symplectic Euler Complete Function
void symplectic_euler_integration(Particle *particles, const int N, const double delta_t);


//Velocity Verlet Integration: 2 parts

//initial 
void velocity_verlet_integration_initial(Particle *particles, const int N, const double delta_t);

//time loop
void velocity_verlet_integration_timeloop(Particle *particles, const int N, const double delta_t);



//----------------------------------------------------------------------------------------------

int main(int argc, char *argv[]) {
    // Check if the correct number of command-line arguments are provided
    if (argc != 6) {
        printf("Give 6 input args: N filename nsteps delta_t graphic\n");
        return -1;
    } 

    // Storing command-line argurments
    const int N = atoi(argv[1]); // Number of particles
    char *filename = argv[2]; // Name of the input file
    const int nsteps = atoi(argv[3]); // Number of time steps
    const double delta_t = atof(argv[4]); // Time step
    int graphics = atoi(argv[5]); // Flag to enable graphics

    // Allocate memory for particles
    Particle *particles = malloc(N * sizeof(Particle));
    if (particles == NULL) {
        printf("Memory allocation failed\n");
        return -1;
    }

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
            free(particles); // Free allocated memory before exiting
            return -1;
        }

        particles[i].position_x = data[0];
        particles[i].position_y = data[1];
        particles[i].mass = data[2];
        particles[i].velocity_x = data[3];
        particles[i].velocity_y = data[4];
        particles[i].brightness = data[5];

        // Initialize computed fields to zero
        particles[i].force_x = 0;  
        particles[i].force_y = 0;
        particles[i].acceleration_x = 0;
        particles[i].acceleration_y = 0;
        particles[i].acceleration_x_buffer = 0;
        particles[i].acceleration_y_buffer = 0;
        //assign particle ID
        particles[i].particle_ID = i;
        //Initialize quadrant
        particles[i].depth = 0;
        particles[i].quadrant = 0;
    }

    fclose(input_file);

    //Assign particle ID 
    for (int i = 0; i < N; i++){
        particles[i].particle_ID = i+1;
    }

    //-----------------------------SIMULATION---------------------


    clock_t start, end;
    start = clock();
     //initial call barnes hut and verlet integrator
    velocity_verlet_integration_initial(particles, N, delta_t);
    //Time Loop
     for (int i = 0; i < nsteps; i++) {
        //timeloop_velocity_verlet function
        velocity_verlet_integration_timeloop(particles, N, delta_t);
        //check if particles are out of bounds
        if (check_out_of_bounds(particles, N) == 1){ 
            return 1;
         }
     }
    end = clock();
    double cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC; // Convert to seconds
    printf("Time taken: %f seconds\n", cpu_time_used); 
    
    //-----------------------END SIMULATION--------------------------------------------------------------------


    char *output_filename = "result.gal";
    FILE *output_file = fopen(output_filename, "wb");
    if (output_file == NULL) {
        printf("Error opening output file: %s\n", output_filename);
        free(particles); // Free allocated memory before exiting
        return -1;
    }

    double data[6];
    for (int i = 0; i < N; i++) {
        data[0] = particles[i].position_x;
        data[1] = particles[i].position_y; 
        data[2] = particles[i].mass;
        data[3] = particles[i].velocity_x;
        data[4] = particles[i].velocity_y;
        data[5] = particles[i].brightness; // Adjust this based on the first 6 fields in your struct

        size_t bytes_written = fwrite(data, sizeof(double), 6, output_file);
        if (bytes_written != 6) {
            printf("Error writing particle %d\n", i);
            fclose(output_file);
            free(particles); // Free allocated memory before exiting
            return -1;
        }
    }

    fclose(output_file);
    free(particles);

    return 0;
}




//quadtree functions 
struct node_qt *create_Node(double xmin, double xmax, double ymin, double ymax, int depth){
    struct node_qt *node = (struct node_qt *) malloc(sizeof(struct node_qt));
    //depth
    node->depth = depth;
    //bounding box
    node->xmin = xmin;
    node->xmax = xmax; 
    node->ymin = ymin;
    node->ymax = ymax;
    //reference to childs
    node->number_of_particles = 0;
    //reference to childs 
    node->ll = NULL;
    node->rl = NULL;
    node->ru = NULL;
    node->lu = NULL;
    node->particle = NULL; 

    node->box_width = node->xmax - node->xmin; 
    node->total_mass = 0.0;
    node->center_of_mass_x = node->center_of_mass_y = 0.0;
    return node;
}



//build quadtree 
void build_quadtree(Particle *particles, int N, struct node_qt **node){
    //loop over all particles
    for (int i = 0; i < N; i++){
        //extract particles
        Particle *particle_i = &particles[i];
        insert_particles(particle_i, node);
    }
    //compute mass and center of mass
    center_of_mass_and_mass(node);
}


//split quadtree
void split_quadtree(struct node_qt **node){

    //Q1 / SW / ll
    (*node)->ll = create_Node((*node)->xmin, ((*node)->xmax + (*node)->xmin)/2, 
                              (*node)->ymin, ((*node)->ymax + (*node)->ymin)/2, (*node)->depth + 1);
    //Q2 / SE / rl
    (*node)->rl = create_Node(((*node)->xmax + (*node)->xmin)/2, (*node)->xmax, 
                               (*node)->ymin, ((*node)->ymax + (*node)->ymin)/2, (*node)->depth + 1);
    //Q3 / NE / ru
    (*node)->ru = create_Node(((*node)->xmax + (*node)->xmin)/2, (*node)->xmax, 
                              ((*node)->ymax + (*node)->ymin)/2, (*node)->ymax, (*node)->depth + 1);
    //Q4 / NW / lu
    (*node)->lu = create_Node((*node)->xmin, ((*node)->xmax + (*node)->xmin)/2, 
                             ((*node)->ymax + (*node)->ymin)/2,(*node)->ymax, (*node)->depth + 1);
    
}


//determine quadrant
int get_quadrant(Particle *particle, struct node_qt **node){
    int quadrant = 0;
        //Q1 
        if (particle->position_x <= ((*node)->xmax + (*node)->xmin)/2 && particle->position_y <= ((*node)->ymax + (*node)->ymin)/2)
            quadrant = 1;
        //Q2
        else if (particle->position_x > ((*node)->xmax + (*node)->xmin)/2 && particle->position_y <= ((*node)->ymax + (*node)->ymin)/2)
            quadrant = 2;
        //Q3
        else if (particle->position_x > ((*node)->xmax + (*node)->xmin)/2 && particle->position_y > ((*node)->ymax + (*node)->ymin)/2)
            quadrant = 3;
        //Q4
        else
            quadrant = 4;
        
        
        return quadrant;
} 

//insert particles and update center of mass
void insert_particles(Particle *particle_i, struct node_qt **node){

    int quad_i = 0; 
    int quad_n = 0;

     // particle end up at same point
    if ((*node)->particle != NULL && (*node)->particle->position_x == particle_i->position_x && (*node)->particle->position_y == particle_i->position_y){
        printf("ERROR: Particles ended up at the same point. Free memory and abort Simulation.\n");
        exit(1);
    }


    //MORE THAN 1 PARTICLE -> I.E. Node has already been created
    else if ((*node)->number_of_particles > 1){ 
        //quadrant of particle i
        quad_i = get_quadrant(particle_i, &(*node));
        //increase number of particle for this node 
        (*node)->number_of_particles += 1;
        //increase depth "travelling throug the levels"
        particle_i->depth+=1;

        //new particle
        //Quadrant I
        if (quad_i == 1){ 
            particle_i->quadrant = 1;
            insert_particles(particle_i, &(*node)->ll);
        }
        //Quadrant II
        if (quad_i == 2){ 
            particle_i->quadrant = 2;
            insert_particles(particle_i, &(*node)->rl);
        }
        //Quadrant III
        if (quad_i == 3){ 
            particle_i->quadrant = 3;
            insert_particles(particle_i, &(*node)->ru);
        }
        //Quadrant IV
        if (quad_i == 4){ 
            particle_i->quadrant = 4;
            insert_particles(particle_i, &(*node)->lu);
        }
    } 


    //1 PARTICLE
    else if ((*node)->number_of_particles == 1) { 
        //paticle in node
        quad_n = get_quadrant((*node)->particle, &(*node));
        //particle i
        quad_i = get_quadrant(particle_i, &(*node));
        //split quadtree
        split_quadtree(node);
        //increase depth due to split
        (*node)->particle->depth+=1;
        particle_i->depth+=1;

        //existing particle
        //Quadrant I

        if (quad_n == 1){
            (*node)->particle->quadrant = 1;
            insert_particles((*node)->particle, &(*node)->ll);  
        }
        //Quadrant II
        if (quad_n == 2){
            (*node)->particle->quadrant = 2;
            insert_particles((*node)->particle, &(*node)->rl);
        }
        //Quadrant III
        if (quad_n == 3){
            (*node)->particle->quadrant = 3;
            insert_particles((*node)->particle, &(*node)->ru);
        }
        //Quadrant IV  
        if (quad_n == 4){
            (*node)->particle->quadrant = 4;
            insert_particles((*node)->particle, &(*node)->lu);
        }


        //remove particle from current node 
        (*node)->particle = NULL; 


        //new particle
        //increase number of particle for this node
       (*node)->number_of_particles += 1;
        //Quadrant I
        if (quad_i == 1){
            particle_i->quadrant = 1;
            insert_particles(particle_i, &(*node)->ll);
        }
        //Quadrant II
        if (quad_i == 2){
            particle_i->quadrant = 2;
            insert_particles(particle_i, &(*node)->rl);
        }
        //Quadrant III
        if (quad_i == 3){
            particle_i->quadrant = 3;
            insert_particles(particle_i, &(*node)->ru);
        }
        //Quadrant IV  
        if (quad_i == 4){
            particle_i->quadrant = 4;
            insert_particles(particle_i, &(*node)->lu);
        }
        return;

    }

    //EMPTY node -> Leaf node
    else if ((*node)->number_of_particles == 0){  
        (*node)->particle = particle_i;
        (*node)->number_of_particles += 1;
        return;
    }

}



//function for both, mass and center of mass 
void center_of_mass_and_mass(struct node_qt **node){

    if ((*node)->number_of_particles == 1){
        (*node)->center_of_mass_x = (*node)->particle->position_x;
        (*node)->center_of_mass_y = (*node)->particle->position_y;
        (*node)->total_mass = (*node)->particle->mass;
        return;
    }

    else if ((*node)->number_of_particles == 0){
        (*node)->center_of_mass_x = 0;
        (*node)->center_of_mass_y = 0;
        (*node)->total_mass = 0; 
        return;
    }
    else{
        center_of_mass_and_mass(&(*node)->ll);
        center_of_mass_and_mass(&(*node)->rl);
        center_of_mass_and_mass(&(*node)->ru);
        center_of_mass_and_mass(&(*node)->lu);
        
        (*node)->total_mass = (*node)->ll->total_mass + (*node)->rl->total_mass + (*node)->ru->total_mass + (*node)->lu->total_mass;
        (*node)->center_of_mass_x = (((*node)->ll->total_mass * (*node)->ll->center_of_mass_x) + ((*node)->rl->total_mass * (*node)->rl->center_of_mass_x) + ((*node)->ru->total_mass * (*node)->ru->center_of_mass_x) + ((*node)->lu->total_mass * (*node)->lu->center_of_mass_x))/(*node)->total_mass;
        (*node)->center_of_mass_y = (((*node)->ll->total_mass * (*node)->ll->center_of_mass_y) + ((*node)->rl->total_mass * (*node)->rl->center_of_mass_y) + ((*node)->ru->total_mass * (*node)->ru->center_of_mass_y) + ((*node)->lu->total_mass * (*node)->lu->center_of_mass_y))/(*node)->total_mass;
    }   
        
}



struct forces calculate_force_qt_ret(Particle *particle_i, struct node_qt **node, const int N, const double theta){

        // Initialize
        const double G = -100.0 / N;
        const double e_0 = 0.001; 
    
        struct forces aux_force = {0,0};
        struct forces aux_force_ll = {0, 0};
        struct forces aux_force_rl = {0, 0};
        struct forces aux_force_ru = {0, 0};
        struct forces aux_force_lu = {0, 0};
      
        //current particle in box is particle i itself
        if ((*node)->particle == particle_i){  
            aux_force.force_x = 0;
            aux_force.force_y = 0;
        }
        //leaf node 
        //contains one particle -> direct interaction between two particles
        else if((*node)->ll == NULL){ 
            const double x_vector =   particle_i->position_x - (*node)->center_of_mass_x;
            const double y_vector =   particle_i->position_y - (*node)->center_of_mass_y;
            const double r = sqrt(x_vector*x_vector + y_vector*y_vector);
            //needed for condition check
            const double r_inv = 1.0/r;
            const double r_epsilon_cubed = (r + e_0) * (r + e_0) * (r + e_0);
            const double r_inv_epsilon_cubed = 1.0/r_epsilon_cubed;

            aux_force.force_x = G * particle_i->mass * (*node)->total_mass * x_vector * r_inv_epsilon_cubed;
            aux_force.force_y = G * particle_i->mass * (*node)->total_mass * y_vector * r_inv_epsilon_cubed;

        }
            
    
    else{
        const double x_vector =   particle_i->position_x - (*node)->center_of_mass_x;
        const double y_vector =   particle_i->position_y - (*node)->center_of_mass_y;
        const double r = sqrt(x_vector*x_vector + y_vector*y_vector);
        //needed for condition check
        const double r_inv = 1.0/r;
        const double r_epsilon_cubed = (r + e_0) * (r + e_0) * (r + e_0);
        const double r_inv_epsilon_cubed = 1.0/r_epsilon_cubed;

        
        if ((*node)->box_width * r_inv < theta) {
            aux_force.force_x = G * particle_i->mass * (*node)->total_mass * x_vector * r_inv_epsilon_cubed;
            aux_force.force_y = G * particle_i->mass * (*node)->total_mass * y_vector * r_inv_epsilon_cubed;
        } 
        //group of particles not sufficiently far away -> break it into subregions; check if children empty
        else {
            if ((*node)->ll != NULL){
                aux_force_ll = calculate_force_qt_ret(particle_i, &(*node)->ll, N, theta);
            }
            if ((*node)->rl != NULL){
                aux_force_rl = calculate_force_qt_ret(particle_i, &(*node)->rl, N, theta);
            }
            if ((*node)->ru != NULL){
                aux_force_ru = calculate_force_qt_ret(particle_i, &(*node)->ru, N, theta);
            }
            if ((*node)->lu != NULL){
                aux_force_lu = calculate_force_qt_ret(particle_i, &(*node)->lu, N, theta);
            }
            //sum 
            aux_force.force_x = aux_force_ll.force_x + aux_force_rl.force_x + aux_force_ru.force_x + aux_force_lu.force_x;
            aux_force.force_y = aux_force_ll.force_y + aux_force_rl.force_y + aux_force_ru.force_y + aux_force_lu.force_y;
        }
    }
       
       
        return aux_force;

    }




//out of bounds check after integration
int check_out_of_bounds(Particle *particles, int N){
   for (int i = 0; i < N; i++){
    Particle *particle = &particles[i];
        if (particle->position_x > 1 || particle->position_x < 0 || particle->position_y > 1 || particle->position_y < 0){
            printf("ERROR: Particle outside the bounding box. Stop simulation.\n");
            return 1;
        }
   }

   return 0; 
}


//free memory
void delete_tree(struct node_qt **node){
    if ((*node) == NULL){
        return;
    }
    else{
        delete_tree(&(*node)->ll);
        delete_tree(&(*node)->rl);
        delete_tree(&(*node)->ru);
        delete_tree(&(*node)->lu);

        free(*node);
        (*node) = NULL; 
    }
}




//print function for debugging purposes
void print_data(struct node_qt **node){
    if ((*node)->number_of_particles == 1 || (*node)->number_of_particles == 0){ 
        printf("Depth: %d, total mass: %.17f: center of mass in x: %f center of mass in y: %f\n", (*node)->depth, (*node)->total_mass, (*node)->center_of_mass_x, (*node)->center_of_mass_y);
    }

    else{
        printf("Depth: %d, total mass: %.15f: center of mass in x: %f center of mass in y: %f\n", (*node)->depth, (*node)->total_mass, (*node)->center_of_mass_x, (*node)->center_of_mass_y);
        print_data(&(*node)->ll);
        print_data(&(*node)->rl);
        print_data(&(*node)->ru);
        print_data(&(*node)->lu);
    }
}




//VELOCITY VERLET COMPONENTS

void velocity_verlet_integration_initial(Particle *particles, const int N, const double delta_t){
        //calculate F(0)
        struct node_qt *root_qt = create_Node(0.0, 1.0, 0.0, 1.0, 0); 
        build_quadtree(particles, N, &root_qt);
        struct forces place_holder_force = {0,0}; 
        for (int i = 0; i < N; i++) { 
            Particle *particle_i = &particles[i]; 
            place_holder_force = calculate_force_qt_ret(particle_i, &root_qt, N, 0.43); 
            particle_i->force_x = place_holder_force.force_x;
            particle_i->force_y = place_holder_force.force_y;
        }
        delete_tree(&root_qt);
        root_qt = NULL; 

    
        //calculate a(0)
        for (int i = 0; i < N; i++) {
            Particle *particle_i = &particles[i];
            //a(t)
            particle_i->acceleration_x = particle_i->force_x/particle_i->mass;
            particle_i->acceleration_y = particle_i->force_y/particle_i->mass;
            //load a(t) into buffer
            particle_i->acceleration_x_buffer = particle_i->acceleration_x;
            particle_i->acceleration_y_buffer = particle_i->acceleration_y;
        }
}

void velocity_verlet_integration_timeloop(Particle *particles, const int N, const double delta_t){
    //Reset depth field and quadrant field in particle struct
    for (int i = 0; i < N; i++){ //VECTORIZED
        particles[i].depth = 0;
        particles[i].quadrant = 0;
        }
  
        //calculate x(t+delta_t)
        for (int i = 0; i < N; i++){ //VECTORIZED
            Particle *particle_i = &particles[i];
            particle_i->position_x += particle_i->velocity_x * delta_t + 0.5 * particle_i->acceleration_x * delta_t * delta_t;
            particle_i->position_y += particle_i->velocity_y * delta_t + 0.5 * particle_i->acceleration_y * delta_t * delta_t;
        }
    
        //calculate F(t+delta_t)
        struct node_qt *root_qt = create_Node(0.0, 1.0, 0.0, 1.0, 0); 
        build_quadtree(particles, N, &root_qt);
        struct forces place_holder_force_f = {0,0}; 
        for (int i = 0; i < N; i++){
            Particle *particle_i = &particles[i];
            place_holder_force_f = calculate_force_qt_ret(particle_i, &root_qt, N, 0.43); 
            particle_i->force_x = place_holder_force_f.force_x;
            particle_i->force_y = place_holder_force_f.force_y;
        }
        delete_tree(&root_qt);
        root_qt = NULL;



        //calculate a(t+delta_t) and v(t+delta_t)
        for (int i = 0; i < N; i++){
            Particle *particle_i = &particles[i];
            //calculate a(t+delta_t)
            particle_i->acceleration_x = particle_i->force_x/particle_i->mass;
            particle_i->acceleration_y = particle_i->force_y/particle_i->mass;

            //calculate v(t+delta_t)
            particle_i->velocity_x += 0.5 * (particle_i->acceleration_x_buffer + particle_i->acceleration_x) * delta_t;
            particle_i->velocity_y += 0.5 * (particle_i->acceleration_y_buffer + particle_i->acceleration_y) * delta_t;
            //load into buffer
            particle_i->acceleration_x_buffer = particle_i->acceleration_x;
            particle_i->acceleration_y_buffer = particle_i->acceleration_y;
        }

}


