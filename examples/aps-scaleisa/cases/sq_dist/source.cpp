extern "C" { float scalar_results[64]; }
static float sq_dist(float a[3],float b[3]) {
  float x=a[0]-b[0],y=a[1]-b[1],z=a[2]-b[2];
  return (x*x+y*y)+z*z;
}
int main() {
  for (int i=0;i<64;++i) {
    float a[3],b[3];
    for (int c=0;c<3;++c) {
      a[c]=(float)((i*37+c*11)%101-50)*0.031f;
      b[c]=(float)((i*19+c*17)%97-48)*0.029f;
    }
    scalar_results[i]=sq_dist(a,b);
  }
  return 0;
}
