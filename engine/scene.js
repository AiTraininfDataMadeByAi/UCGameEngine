'use strict';
// ============================================================
//  UCGameEngine — Renderer v0.1.0
//  Full WebGL 1.0 renderer built from scratch.
//  No libraries. No dependencies. Pure WebGL.
// ============================================================

// ============================================================
//  SECTION 1 — MATH LIBRARY
//  Vec2, Vec3, Vec4, Mat4, Quaternion
// ============================================================

const UCMath = {

  // --- Constants ---
  DEG2RAD: Math.PI / 180,
  RAD2DEG: 180 / Math.PI,
  PI:      Math.PI,
  TAU:     Math.PI * 2,
  EPSILON: 0.000001,

  clamp(v, min, max) { return Math.max(min, Math.min(max, v)); },
  lerp(a, b, t)      { return a + (b - a) * t; },
  smoothstep(e0, e1, x) {
    const t = this.clamp((x - e0) / (e1 - e0), 0, 1);
    return t * t * (3 - 2 * t);
  },

  // ---- Vec3 ----
  Vec3: {
    create(x=0,y=0,z=0)   { return new Float32Array([x,y,z]); },
    clone(v)               { return new Float32Array([v[0],v[1],v[2]]); },
    copy(out,v)            { out[0]=v[0];out[1]=v[1];out[2]=v[2];return out; },
    set(out,x,y,z)         { out[0]=x;out[1]=y;out[2]=z;return out; },
    add(out,a,b)           { out[0]=a[0]+b[0];out[1]=a[1]+b[1];out[2]=a[2]+b[2];return out; },
    sub(out,a,b)           { out[0]=a[0]-b[0];out[1]=a[1]-b[1];out[2]=a[2]-b[2];return out; },
    scale(out,a,s)         { out[0]=a[0]*s;out[1]=a[1]*s;out[2]=a[2]*s;return out; },
    mul(out,a,b)           { out[0]=a[0]*b[0];out[1]=a[1]*b[1];out[2]=a[2]*b[2];return out; },
    dot(a,b)               { return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]; },
    cross(out,a,b)         {
      const ax=a[0],ay=a[1],az=a[2],bx=b[0],by=b[1],bz=b[2];
      out[0]=ay*bz-az*by; out[1]=az*bx-ax*bz; out[2]=ax*by-ay*bx;
      return out;
    },
    len(v)                 { return Math.sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]); },
    lenSq(v)               { return v[0]*v[0]+v[1]*v[1]+v[2]*v[2]; },
    normalize(out,v)       {
      let l = Math.sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]);
      if (l < 0.000001) { out[0]=0;out[1]=0;out[2]=0; return out; }
      l = 1/l;
      out[0]=v[0]*l; out[1]=v[1]*l; out[2]=v[2]*l;
      return out;
    },
    negate(out,v)          { out[0]=-v[0];out[1]=-v[1];out[2]=-v[2];return out; },
    dist(a,b)              {
      const dx=a[0]-b[0],dy=a[1]-b[1],dz=a[2]-b[2];
      return Math.sqrt(dx*dx+dy*dy+dz*dz);
    },
    lerp(out,a,b,t)        {
      out[0]=a[0]+(b[0]-a[0])*t;
      out[1]=a[1]+(b[1]-a[1])*t;
      out[2]=a[2]+(b[2]-a[2])*t;
      return out;
    },
    transformMat4(out,v,m) {
      const x=v[0],y=v[1],z=v[2];
      let w = m[3]*x+m[7]*y+m[11]*z+m[15];
      w = w || 1;
      out[0]=(m[0]*x+m[4]*y+m[8] *z+m[12])/w;
      out[1]=(m[1]*x+m[5]*y+m[9] *z+m[13])/w;
      out[2]=(m[2]*x+m[6]*y+m[10]*z+m[14])/w;
      return out;
    },
    str(v) { return `(${v[0].toFixed(2)}, ${v[1].toFixed(2)}, ${v[2].toFixed(2)})`; }
  },

  // ---- Vec4 ----
  Vec4: {
    create(x=0,y=0,z=0,w=1) { return new Float32Array([x,y,z,w]); },
    clone(v)                  { return new Float32Array([v[0],v[1],v[2],v[3]]); },
    transformMat4(out,v,m)   {
      out[0]=m[0]*v[0]+m[4]*v[1]+m[8] *v[2]+m[12]*v[3];
      out[1]=m[1]*v[0]+m[5]*v[1]+m[9] *v[2]+m[13]*v[3];
      out[2]=m[2]*v[0]+m[6]*v[1]+m[10]*v[2]+m[14]*v[3];
      out[3]=m[3]*v[0]+m[7]*v[1]+m[11]*v[2]+m[15]*v[3];
      return out;
    }
  },

  // ---- Mat4 ----
  // Column-major, matching WebGL convention
  Mat4: {
    create() {
      const m = new Float32Array(16);
      m[0]=1; m[5]=1; m[10]=1; m[15]=1;
      return m;
    },
    clone(a)       { return new Float32Array(a); },
    identity(out)  {
      out.fill(0);
      out[0]=1; out[5]=1; out[10]=1; out[15]=1;
      return out;
    },
    copy(out,a)    { for(let i=0;i<16;i++) out[i]=a[i]; return out; },
    multiply(out,a,b) {
      const a00=a[0],a01=a[1],a02=a[2],a03=a[3];
      const a10=a[4],a11=a[5],a12=a[6],a13=a[7];
      const a20=a[8],a21=a[9],a22=a[10],a23=a[11];
      const a30=a[12],a31=a[13],a32=a[14],a33=a[15];
      let b0=b[0],b1=b[1],b2=b[2],b3=b[3];
      out[0]=b0*a00+b1*a10+b2*a20+b3*a30;
      out[1]=b0*a01+b1*a11+b2*a21+b3*a31;
      out[2]=b0*a02+b1*a12+b2*a22+b3*a32;
      out[3]=b0*a03+b1*a13+b2*a23+b3*a33;
      b0=b[4];b1=b[5];b2=b[6];b3=b[7];
      out[4]=b0*a00+b1*a10+b2*a20+b3*a30;
      out[5]=b0*a01+b1*a11+b2*a21+b3*a31;
      out[6]=b0*a02+b1*a12+b2*a22+b3*a32;
      out[7]=b0*a03+b1*a13+b2*a23+b3*a33;
      b0=b[8];b1=b[9];b2=b[10];b3=b[11];
      out[8] =b0*a00+b1*a10+b2*a20+b3*a30;
      out[9] =b0*a01+b1*a11+b2*a21+b3*a31;
      out[10]=b0*a02+b1*a12+b2*a22+b3*a32;
      out[11]=b0*a03+b1*a13+b2*a23+b3*a33;
      b0=b[12];b1=b[13];b2=b[14];b3=b[15];
      out[12]=b0*a00+b1*a10+b2*a20+b3*a30;
      out[13]=b0*a01+b1*a11+b2*a21+b3*a31;
      out[14]=b0*a02+b1*a12+b2*a22+b3*a32;
      out[15]=b0*a03+b1*a13+b2*a23+b3*a33;
      return out;
    },
    translate(out,a,v) {
      const x=v[0],y=v[1],z=v[2];
      if (a === out) {
        out[12]=a[0]*x+a[4]*y+a[8]*z+a[12];
        out[13]=a[1]*x+a[5]*y+a[9]*z+a[13];
        out[14]=a[2]*x+a[6]*y+a[10]*z+a[14];
        out[15]=a[3]*x+a[7]*y+a[11]*z+a[15];
      } else {
        for(let i=0;i<12;i++) out[i]=a[i];
        out[12]=a[0]*x+a[4]*y+a[8]*z+a[12];
        out[13]=a[1]*x+a[5]*y+a[9]*z+a[13];
        out[14]=a[2]*x+a[6]*y+a[10]*z+a[14];
        out[15]=a[3]*x+a[7]*y+a[11]*z+a[15];
      }
      return out;
    },
    scale(out,a,v) {
      const x=v[0],y=v[1],z=v[2];
      out[0]=a[0]*x; out[1]=a[1]*x; out[2]=a[2]*x;  out[3]=a[3]*x;
      out[4]=a[4]*y; out[5]=a[5]*y; out[6]=a[6]*y;  out[7]=a[7]*y;
      out[8]=a[8]*z; out[9]=a[9]*z; out[10]=a[10]*z; out[11]=a[11]*z;
      out[12]=a[12]; out[13]=a[13]; out[14]=a[14];   out[15]=a[15];
      return out;
    },
    rotateX(out,a,rad) {
      const s=Math.sin(rad),c=Math.cos(rad);
      const a10=a[4],a11=a[5],a12=a[6],a13=a[7];
      const a20=a[8],a21=a[9],a22=a[10],a23=a[11];
      if (a!==out){for(let i=0;i<4;i++)out[i]=a[i];for(let i=12;i<16;i++)out[i]=a[i];}
      out[4]=a10*c+a20*s; out[5]=a11*c+a21*s;
      out[6]=a12*c+a22*s; out[7]=a13*c+a23*s;
      out[8]=a20*c-a10*s; out[9]=a21*c-a11*s;
      out[10]=a22*c-a12*s;out[11]=a23*c-a13*s;
      return out;
    },
    rotateY(out,a,rad) {
      const s=Math.sin(rad),c=Math.cos(rad);
      const a00=a[0],a01=a[1],a02=a[2],a03=a[3];
      const a20=a[8],a21=a[9],a22=a[10],a23=a[11];
      if (a!==out){for(let i=4;i<8;i++)out[i]=a[i];for(let i=12;i<16;i++)out[i]=a[i];}
      out[0]=a00*c-a20*s; out[1]=a01*c-a21*s;
      out[2]=a02*c-a22*s; out[3]=a03*c-a23*s;
      out[8]=a00*s+a20*c; out[9]=a01*s+a21*c;
      out[10]=a02*s+a22*c;out[11]=a03*s+a23*c;
      return out;
    },
    rotateZ(out,a,rad) {
      const s=Math.sin(rad),c=Math.cos(rad);
      const a00=a[0],a01=a[1],a02=a[2],a03=a[3];
      const a10=a[4],a11=a[5],a12=a[6],a13=a[7];
      if (a!==out){for(let i=8;i<16;i++)out[i]=a[i];}
      out[0]=a00*c+a10*s; out[1]=a01*c+a11*s;
      out[2]=a02*c+a12*s; out[3]=a03*c+a13*s;
      out[4]=a10*c-a00*s; out[5]=a11*c-a01*s;
      out[6]=a12*c-a02*s; out[7]=a13*c-a03*s;
      return out;
    },
    fromRotationTranslationScale(out,q,v,s) {
      const x=q[0],y=q[1],z=q[2],w=q[3];
      const x2=x+x,y2=y+y,z2=z+z;
      const xx=x*x2,xy=x*y2,xz=x*z2;
      const yy=y*y2,yz=y*z2,zz=z*z2;
      const wx=w*x2,wy=w*y2,wz=w*z2;
      const sx=s[0],sy=s[1],sz=s[2];
      out[0]=(1-(yy+zz))*sx; out[1]=(xy+wz)*sx;    out[2]=(xz-wy)*sx;    out[3]=0;
      out[4]=(xy-wz)*sy;    out[5]=(1-(xx+zz))*sy; out[6]=(yz+wx)*sy;    out[7]=0;
      out[8]=(xz+wy)*sz;    out[9]=(yz-wx)*sz;     out[10]=(1-(xx+yy))*sz;out[11]=0;
      out[12]=v[0];          out[13]=v[1];           out[14]=v[2];          out[15]=1;
      return out;
    },
    perspective(out,fovy,aspect,near,far) {
      const f=1/Math.tan(fovy/2);
      const nf = far === Infinity ? -1 : 1/(near-far);
      out[0]=f/aspect; out[1]=0; out[2]=0;              out[3]=0;
      out[4]=0;        out[5]=f; out[6]=0;              out[7]=0;
      out[8]=0;        out[9]=0; out[10]=(far+near)*nf; out[11]=-1;
      out[12]=0;       out[13]=0;out[14]=2*far*near*nf; out[15]=0;
      return out;
    },
    ortho(out,left,right,bottom,top,near,far) {
      const lr=1/(left-right),bt=1/(bottom-top),nf=1/(near-far);
      out[0]=-2*lr;      out[1]=0;       out[2]=0;     out[3]=0;
      out[4]=0;          out[5]=-2*bt;   out[6]=0;     out[7]=0;
      out[8]=0;          out[9]=0;       out[10]=2*nf; out[11]=0;
      out[12]=(left+right)*lr;
      out[13]=(top+bottom)*bt;
      out[14]=(far+near)*nf;
      out[15]=1;
      return out;
    },
    lookAt(out,eye,center,up) {
      const V3 = UCMath.Vec3;
      let z0=eye[0]-center[0],z1=eye[1]-center[1],z2=eye[2]-center[2];
      let len=z0*z0+z1*z1+z2*z2;
      if(len>0){len=1/Math.sqrt(len);z0*=len;z1*=len;z2*=len;}
      let x0=up[1]*z2-up[2]*z1,x1=up[2]*z0-up[0]*z2,x2=up[0]*z1-up[1]*z0;
      len=x0*x0+x1*x1+x2*x2;
      if(len>0){len=1/Math.sqrt(len);x0*=len;x1*=len;x2*=len;}
      const y0=z1*x2-z2*x1,y1=z2*x0-z0*x2,y2=z0*x1-z1*x0;
      out[0]=x0;out[1]=y0;out[2]=z0;out[3]=0;
      out[4]=x1;out[5]=y1;out[6]=z1;out[7]=0;
      out[8]=x2;out[9]=y2;out[10]=z2;out[11]=0;
      out[12]=-(x0*eye[0]+x1*eye[1]+x2*eye[2]);
      out[13]=-(y0*eye[0]+y1*eye[1]+y2*eye[2]);
      out[14]=-(z0*eye[0]+z1*eye[1]+z2*eye[2]);
      out[15]=1;
      return out;
    },
    invert(out,a) {
      const a00=a[0],a01=a[1],a02=a[2],a03=a[3];
      const a10=a[4],a11=a[5],a12=a[6],a13=a[7];
      const a20=a[8],a21=a[9],a22=a[10],a23=a[11];
      const a30=a[12],a31=a[13],a32=a[14],a33=a[15];
      const b00=a00*a11-a01*a10,b01=a00*a12-a02*a10;
      const b02=a00*a13-a03*a10,b03=a01*a12-a02*a11;
      const b04=a01*a13-a03*a11,b05=a02*a13-a03*a12;
      const b06=a20*a31-a21*a30,b07=a20*a32-a22*a30;
      const b08=a20*a33-a23*a30,b09=a21*a32-a22*a31;
      const b10=a21*a33-a23*a31,b11=a22*a33-a23*a32;
      let det=b00*b11-b01*b10+b02*b09+b03*b08-b04*b07+b05*b06;
      if(!det) return null;
      det=1/det;
      out[0]=(a11*b11-a12*b10+a13*b09)*det;
      out[1]=(a02*b10-a01*b11-a03*b09)*det;
      out[2]=(a31*b05-a32*b04+a33*b03)*det;
      out[3]=(a22*b04-a21*b05-a23*b03)*det;
      out[4]=(a12*b08-a10*b11-a13*b07)*det;
      out[5]=(a00*b11-a02*b08+a03*b07)*det;
      out[6]=(a32*b02-a30*b05-a33*b01)*det;
      out[7]=(a20*b05-a22*b02+a23*b01)*det;
      out[8]=(a10*b10-a11*b08+a13*b06)*det;
      out[9]=(a01*b08-a00*b10-a03*b06)*det;
      out[10]=(a30*b04-a31*b02+a33*b00)*det;
      out[11]=(a21*b02-a20*b04-a23*b00)*det;
      out[12]=(a11*b07-a10*b09-a12*b06)*det;
      out[13]=(a00*b09-a01*b07+a02*b06)*det;
      out[14]=(a31*b01-a30*b03-a32*b00)*det;
      out[15]=(a20*b03-a21*b01+a22*b00)*det;
      return out;
    },
    transpose(out,a) {
      if(out===a){
        let t;
        t=a[1]; out[1]=a[4];  out[4]=t;
        t=a[2]; out[2]=a[8];  out[8]=t;
        t=a[3]; out[3]=a[12]; out[12]=t;
        t=a[6]; out[6]=a[9];  out[9]=t;
        t=a[7]; out[7]=a[13]; out[13]=t;
        t=a[11];out[11]=a[14];out[14]=t;
      } else {
        out[0]=a[0];  out[1]=a[4];  out[2]=a[8];  out[3]=a[12];
        out[4]=a[1];  out[5]=a[5];  out[6]=a[9];  out[7]=a[13];
        out[8]=a[2];  out[9]=a[6];  out[10]=a[10];out[11]=a[14];
        out[12]=a[3]; out[13]=a[7]; out[14]=a[11];out[15]=a[15];
      }
      return out;
    },
    normalMatrix(out3,a) {
      // Extract 3x3 normal matrix (inverse transpose of upper-left 3x3)
      const tmp = UCMath.Mat4.create();
      UCMath.Mat4.invert(tmp, a);
      out3[0]=tmp[0];out3[1]=tmp[4];out3[2]=tmp[8];
      out3[3]=tmp[1];out3[4]=tmp[5];out3[5]=tmp[9];
      out3[6]=tmp[2];out3[7]=tmp[6];out3[8]=tmp[10];
      return out3;
    }
  },

  // ---- Quaternion ----
  Quat: {
    create()           { return new Float32Array([0,0,0,1]); },
    identity(out)      { out[0]=0;out[1]=0;out[2]=0;out[3]=1;return out; },
    clone(q)           { return new Float32Array(q); },
    fromEuler(out,x,y,z) {
      const hx=x*UCMath.DEG2RAD*0.5, hy=y*UCMath.DEG2RAD*0.5, hz=z*UCMath.DEG2RAD*0.5;
      const sx=Math.sin(hx),cx=Math.cos(hx);
      const sy=Math.sin(hy),cy=Math.cos(hy);
      const sz=Math.sin(hz),cz=Math.cos(hz);
      out[0]=sx*cy*cz+cx*sy*sz;
      out[1]=cx*sy*cz-sx*cy*sz;
      out[2]=cx*cy*sz+sx*sy*cz;
      out[3]=cx*cy*cz-sx*sy*sz;
      return out;
    },
    multiply(out,a,b) {
      const ax=a[0],ay=a[1],az=a[2],aw=a[3];
      const bx=b[0],by=b[1],bz=b[2],bw=b[3];
      out[0]=ax*bw+aw*bx+ay*bz-az*by;
      out[1]=ay*bw+aw*by+az*bx-ax*bz;
      out[2]=az*bw+aw*bz+ax*by-ay*bx;
      out[3]=aw*bw-ax*bx-ay*by-az*bz;
      return out;
    },
    normalize(out,q) {
      let l=q[0]*q[0]+q[1]*q[1]+q[2]*q[2]+q[3]*q[3];
      if(l>0){l=1/Math.sqrt(l);}
      out[0]=q[0]*l;out[1]=q[1]*l;out[2]=q[2]*l;out[3]=q[3]*l;
      return out;
    },
    slerp(out,a,b,t) {
      let ax=a[0],ay=a[1],az=a[2],aw=a[3];
      let bx=b[0],by=b[1],bz=b[2],bw=b[3];
      let omega,cosom,sinom,scale0,scale1;
      cosom=ax*bx+ay*by+az*bz+aw*bw;
      if(cosom<0){cosom=-cosom;bx=-bx;by=-by;bz=-bz;bw=-bw;}
      if(1-cosom>0.000001){
        omega=Math.acos(cosom);sinom=Math.sin(omega);
        scale0=Math.sin((1-t)*omega)/sinom;
        scale1=Math.sin(t*omega)/sinom;
      } else {
        scale0=1-t; scale1=t;
      }
      out[0]=scale0*ax+scale1*bx;out[1]=scale0*ay+scale1*by;
      out[2]=scale0*az+scale1*bz;out[3]=scale0*aw+scale1*bw;
      return out;
    }
  }
};

// ============================================================
//  SECTION 2 — SHADER SOURCES
// ============================================================

const UCShaders = {

  // -- Standard Mesh Vertex Shader --
  MESH_VERT: `
    attribute vec3 a_position;
    attribute vec3 a_normal;
    attribute vec2 a_uv;

    uniform mat4 u_modelMatrix;
    uniform mat4 u_viewMatrix;
    uniform mat4 u_projMatrix;
    uniform mat3 u_normalMatrix;

    varying vec3 v_worldPos;
    varying vec3 v_normal;
    varying vec2 v_uv;

    void main() {
      vec4 worldPos  = u_modelMatrix * vec4(a_position, 1.0);
      v_worldPos     = worldPos.xyz;
      v_normal       = normalize(u_normalMatrix * a_normal);
      v_uv           = a_uv;
      gl_Position    = u_projMatrix * u_viewMatrix * worldPos;
    }
  `,

  // -- Standard Mesh Fragment Shader --
  MESH_FRAG: `
    precision mediump float;

    varying vec3 v_worldPos;
    varying vec3 v_normal;
    varying vec2 v_uv;

    uniform vec3  u_baseColor;
    uniform float u_metallic;
    uniform float u_roughness;
    uniform float u_emission;
    uniform int   u_useTexture;
    uniform sampler2D u_albedoTex;

    // Lights
    uniform vec3  u_ambientColor;
    uniform float u_ambientIntensity;

    uniform vec3  u_dirLightDir;
    uniform vec3  u_dirLightColor;
    uniform float u_dirLightIntensity;

    uniform vec3  u_pointLight0Pos;
    uniform vec3  u_pointLight0Color;
    uniform float u_pointLight0Intensity;
    uniform float u_pointLight0Range;

    uniform vec3  u_cameraPos;

    vec3 fresnelSchlick(float cosTheta, vec3 F0) {
      return F0 + (1.0 - F0) * pow(max(1.0 - cosTheta, 0.0), 5.0);
    }

    float distributionGGX(vec3 N, vec3 H, float roughness) {
      float a  = roughness * roughness;
      float a2 = a * a;
      float NdH  = max(dot(N, H), 0.0);
      float NdH2 = NdH * NdH;
      float denom = NdH2 * (a2 - 1.0) + 1.0;
      return a2 / (3.14159265 * denom * denom);
    }

    float geometrySmith(float NdV, float NdL, float roughness) {
      float r = roughness + 1.0;
      float k = (r * r) / 8.0;
      float g1 = NdV / (NdV * (1.0 - k) + k);
      float g2 = NdL / (NdL * (1.0 - k) + k);
      return g1 * g2;
    }

    vec3 calcDirLight(vec3 N, vec3 V, vec3 albedo) {
      vec3  L   = normalize(-u_dirLightDir);
      vec3  H   = normalize(V + L);
      float NdL = max(dot(N, L), 0.0);
      float NdV = max(dot(N, V), 0.0);
      vec3  F0  = mix(vec3(0.04), albedo, u_metallic);
      vec3  F   = fresnelSchlick(max(dot(H, V), 0.0), F0);
      float D   = distributionGGX(N, H, u_roughness);
      float G   = geometrySmith(NdV, NdL, u_roughness);
      vec3  spec = (D * G * F) / max(4.0 * NdV * NdL, 0.001);
      vec3  kD  = (vec3(1.0) - F) * (1.0 - u_metallic);
      return (kD * albedo / 3.14159265 + spec) * u_dirLightColor * u_dirLightIntensity * NdL;
    }

    vec3 calcPointLight(vec3 N, vec3 V, vec3 albedo, vec3 lightPos,
                        vec3 lightColor, float intensity, float range) {
      vec3  dir  = lightPos - v_worldPos;
      float dist = length(dir);
      if (dist > range) return vec3(0.0);
      vec3  L    = normalize(dir);
      vec3  H    = normalize(V + L);
      float NdL  = max(dot(N, L), 0.0);
      float NdV  = max(dot(N, V), 0.0);
      float att  = 1.0 - smoothstep(0.0, range, dist);
      att        = att * att;
      vec3  F0   = mix(vec3(0.04), albedo, u_metallic);
      vec3  F    = fresnelSchlick(max(dot(H, V), 0.0), F0);
      float D    = distributionGGX(N, H, u_roughness);
      float G    = geometrySmith(NdV, NdL, u_roughness);
      vec3  spec = (D * G * F) / max(4.0 * NdV * NdL, 0.001);
      vec3  kD   = (vec3(1.0) - F) * (1.0 - u_metallic);
      return (kD * albedo / 3.14159265 + spec) * lightColor * intensity * NdL * att;
    }

    void main() {
      vec3 albedo = u_baseColor;
      if (u_useTexture == 1) {
        albedo = albedo * texture2D(u_albedoTex, v_uv).rgb;
      }
      vec3 N = normalize(v_normal);
      vec3 V = normalize(u_cameraPos - v_worldPos);

      vec3 ambient = u_ambientColor * u_ambientIntensity * albedo;
      vec3 color   = ambient;
      color += calcDirLight(N, V, albedo);
      color += calcPointLight(N, V, albedo,
               u_pointLight0Pos, u_pointLight0Color,
               u_pointLight0Intensity, u_pointLight0Range);
      color += albedo * u_emission;

      // Tone mapping (Reinhard)
      color = color / (color + vec3(1.0));
      // Gamma correction
      color = pow(color, vec3(1.0 / 2.2));

      gl_FragColor = vec4(color, 1.0);
    }
  `,

  // -- Wireframe / Unlit Vertex Shader --
  UNLIT_VERT: `
    attribute vec3 a_position;
    uniform mat4 u_modelMatrix;
    uniform mat4 u_viewMatrix;
    uniform mat4 u_projMatrix;
    void main() {
      gl_Position = u_projMatrix * u_viewMatrix * u_modelMatrix * vec4(a_position, 1.0);
    }
  `,

  // -- Wireframe / Unlit Fragment Shader --
  UNLIT_FRAG: `
    precision mediump float;
    uniform vec3  u_color;
    uniform float u_alpha;
    void main() {
      gl_FragColor = vec4(u_color, u_alpha);
    }
  `,

  // -- Grid Vertex Shader --
  GRID_VERT: `
    attribute vec3 a_position;
    uniform mat4 u_viewMatrix;
    uniform mat4 u_projMatrix;
    uniform vec3 u_cameraPos;
    varying vec3 v_worldPos;
    varying float v_dist;
    void main() {
      vec3 pos = a_position + vec3(u_cameraPos.x, 0.0, u_cameraPos.z);
      v_worldPos = pos;
      v_dist = length(pos - u_cameraPos);
      gl_Position = u_projMatrix * u_viewMatrix * vec4(pos, 1.0);
    }
  `,

  // -- Grid Fragment Shader --
  GRID_FRAG: `
    precision mediump float;
    varying vec3  v_worldPos;
    varying float v_dist;
    uniform float u_gridSize;
    uniform float u_lineWidth;
    uniform vec3  u_gridColor;
    uniform float u_fadeDistance;

    float grid(vec2 p, float s) {
      vec2 g = abs(fract(p / s - 0.5) - 0.5) / fwidth(p / s);
      return min(g.x, g.y);
    }

    void main() {
      float g1 = grid(v_worldPos.xz, 1.0);
      float g10 = grid(v_worldPos.xz, 10.0);
      float line1  = 1.0 - min(g1,  u_lineWidth);
      float line10 = 1.0 - min(g10, u_lineWidth * 2.0);
      float alpha = max(line1 * 0.25, line10 * 0.6);
      float fade = 1.0 - smoothstep(u_fadeDistance * 0.5, u_fadeDistance, v_dist);
      // Axes
      float axisX = 1.0 - min(abs(v_worldPos.z) / fwidth(v_worldPos.z), 1.0);
      float axisZ = 1.0 - min(abs(v_worldPos.x) / fwidth(v_worldPos.x), 1.0);
      vec3 col = u_gridColor;
      if (axisZ > 0.5) col = vec3(0.8, 0.15, 0.15);
      if (axisX > 0.5) col = vec3(0.15, 0.15, 0.8);
      alpha = max(alpha, max(axisX, axisZ) * 0.9);
      gl_FragColor = vec4(col, alpha * fade);
    }
  `,

  // -- Selection Outline Vertex Shader --
  OUTLINE_VERT: `
    attribute vec3 a_position;
    attribute vec3 a_normal;
    uniform mat4 u_modelMatrix;
    uniform mat4 u_viewMatrix;
    uniform mat4 u_projMatrix;
    uniform float u_outlineWidth;
    void main() {
      vec3 pos    = a_position + a_normal * u_outlineWidth;
      gl_Position = u_projMatrix * u_viewMatrix * u_modelMatrix * vec4(pos, 1.0);
    }
  `,

  // -- Selection Outline Fragment Shader --
  OUTLINE_FRAG: `
    precision mediump float;
    uniform vec3 u_outlineColor;
    void main() {
      gl_FragColor = vec4(u_outlineColor, 1.0);
    }
  `,

  // -- Skybox Vertex Shader --
  SKY_VERT: `
    attribute vec3 a_position;
    uniform mat4 u_viewMatrix;
    uniform mat4 u_projMatrix;
    varying vec3 v_dir;
    void main() {
      mat4 view = u_viewMatrix;
      // Remove translation from view matrix
      view[3][0] = 0.0; view[3][1] = 0.0; view[3][2] = 0.0;
      vec4 pos = u_projMatrix * view * vec4(a_position, 1.0);
      gl_Position = pos.xyww;
      v_dir = a_position;
    }
  `,

  // -- Skybox Fragment Shader --
  SKY_FRAG: `
    precision mediump float;
    varying vec3 v_dir;
    uniform vec3 u_skyTop;
    uniform vec3 u_skyHorizon;
    uniform vec3 u_skyBottom;
    void main() {
      float t = clamp(v_dir.y * 0.5 + 0.5, 0.0, 1.0);
      vec3 col = t > 0.5
        ? mix(u_skyHorizon, u_skyTop,    (t - 0.5) * 2.0)
        : mix(u_skyBottom,  u_skyHorizon, t * 2.0);
      gl_FragColor = vec4(col, 1.0);
    }
  `
};

// ============================================================
//  SECTION 3 — GEOMETRY BUILDERS
//  Returns { vertices, normals, uvs, indices }
// ============================================================

const UCGeometry = {

  createCube(size = 1) {
    const h = size / 2;
    const positions = new Float32Array([
      // +X  face
       h,-h,-h,  h, h,-h,  h, h, h,  h,-h, h,
      // -X face
      -h,-h, h, -h, h, h, -h, h,-h, -h,-h,-h,
      // +Y face
      -h, h,-h, -h, h, h,  h, h, h,  h, h,-h,
      // -Y face
      -h,-h, h, -h,-h,-h,  h,-h,-h,  h,-h, h,
      // +Z face
      -h,-h, h,  h,-h, h,  h, h, h, -h, h, h,
      // -Z face
       h,-h,-h, -h,-h,-h, -h, h,-h,  h, h,-h,
    ]);
    const n = [ 1,0,0, -1,0,0, 0,1,0, 0,-1,0, 0,0,1, 0,0,-1 ];
    const normals = new Float32Array(24 * 3);
    for (let f = 0; f < 6; f++) {
      for (let v = 0; v < 4; v++) {
        normals[(f*4+v)*3+0] = n[f*3+0];
        normals[(f*4+v)*3+1] = n[f*3+1];
        normals[(f*4+v)*3+2] = n[f*3+2];
      }
    }
    const uvCoords = new Float32Array([
      0,0, 0,1, 1,1, 1,0,
      0,0, 0,1, 1,1, 1,0,
      0,0, 0,1, 1,1, 1,0,
      0,0, 0,1, 1,1, 1,0,
      0,0, 1,0, 1,1, 0,1,
      0,0, 1,0, 1,1, 0,1,
    ]);
    const indices = new Uint16Array(36);
    for (let f = 0; f < 6; f++) {
      const b = f * 4;
      const i = f * 6;
      indices[i+0]=b+0; indices[i+1]=b+1; indices[i+2]=b+2;
      indices[i+3]=b+0; indices[i+4]=b+2; indices[i+5]=b+3;
    }
    return { positions, normals, uvs: uvCoords, indices };
  },

  createSphere(radius = 0.5, stacks = 24, slices = 24) {
    const vCount = (stacks + 1) * (slices + 1);
    const positions = new Float32Array(vCount * 3);
    const normals   = new Float32Array(vCount * 3);
    const uvs       = new Float32Array(vCount * 2);
    let vi = 0, ni = 0, ui = 0;
    for (let st = 0; st <= stacks; st++) {
      const phi   = Math.PI * st / stacks;
      const sinPhi = Math.sin(phi);
      const cosPhi = Math.cos(phi);
      for (let sl = 0; sl <= slices; sl++) {
        const theta    = Math.PI * 2 * sl / slices;
        const sinTheta = Math.sin(theta);
        const cosTheta = Math.cos(theta);
        const x = cosTheta * sinPhi;
        const y = cosPhi;
        const z = sinTheta * sinPhi;
        positions[vi++] = x * radius;
        positions[vi++] = y * radius;
        positions[vi++] = z * radius;
        normals[ni++] = x; normals[ni++] = y; normals[ni++] = z;
        uvs[ui++] = sl / slices;
        uvs[ui++] = st / stacks;
      }
    }
    const indexCount = stacks * slices * 6;
    const indices    = new Uint16Array(indexCount);
    let idx = 0;
    for (let st = 0; st < stacks; st++) {
      for (let sl = 0; sl < slices; sl++) {
        const a = st * (slices + 1) + sl;
        const b = a + (slices + 1);
        indices[idx++]=a;   indices[idx++]=b;   indices[idx++]=a+1;
        indices[idx++]=b;   indices[idx++]=b+1; indices[idx++]=a+1;
      }
    }
    return { positions, normals, uvs, indices };
  },

  createPlane(width = 10, depth = 10, segsW = 1, segsD = 1) {
    const vW = segsW + 1, vD = segsD + 1;
    const positions = new Float32Array(vW * vD * 3);
    const normals   = new Float32Array(vW * vD * 3);
    const uvs       = new Float32Array(vW * vD * 2);
    let vi=0, ni=0, ui=0;
    for (let iz = 0; iz < vD; iz++) {
      for (let ix = 0; ix < vW; ix++) {
        const x = (ix / segsW - 0.5) * width;
        const z = (iz / segsD - 0.5) * depth;
        positions[vi++]=x; positions[vi++]=0; positions[vi++]=z;
        normals[ni++]=0;   normals[ni++]=1;   normals[ni++]=0;
        uvs[ui++]=ix/segsW; uvs[ui++]=iz/segsD;
      }
    }
    const indices = new Uint16Array(segsW * segsD * 6);
    let idx=0;
    for (let iz=0; iz<segsD; iz++) {
      for (let ix=0; ix<segsW; ix++) {
        const a=iz*vW+ix, b=a+vW;
        indices[idx++]=a;   indices[idx++]=b;   indices[idx++]=a+1;
        indices[idx++]=b;   indices[idx++]=b+1; indices[idx++]=a+1;
      }
    }
    return { positions, normals, uvs, indices };
  },

  createCylinder(radiusTop=0.5, radiusBottom=0.5, height=1, radialSegs=20, heightSegs=1) {
    const positions=[], normals=[], uvs=[], indices=[];
    const halfH = height / 2;
    // Body
    for (let y=0; y<=heightSegs; y++) {
      const t   = y / heightSegs;
      const r   = radiusBottom + (radiusTop - radiusBottom) * t;
      const yPos= -halfH + height * t;
      for (let x=0; x<=radialSegs; x++) {
        const theta  = x / radialSegs * Math.PI * 2;
        const sin    = Math.sin(theta), cos = Math.cos(theta);
        const slope  = (radiusBottom - radiusTop) / height;
        const nLen   = Math.sqrt(1 + slope * slope);
        positions.push(cos*r, yPos, sin*r);
        normals.push(cos/nLen, slope/nLen, sin/nLen);
        uvs.push(x/radialSegs, t);
      }
    }
    const stride = radialSegs + 1;
    for (let y=0; y<heightSegs; y++) {
      for (let x=0; x<radialSegs; x++) {
        const a=y*stride+x, b=(y+1)*stride+x;
        indices.push(a,b,a+1, b,b+1,a+1);
      }
    }
    // Top cap
    const topIdx = positions.length / 3;
    positions.push(0, halfH, 0); normals.push(0,1,0); uvs.push(0.5,0.5);
    for (let x=0; x<=radialSegs; x++) {
      const theta=x/radialSegs*Math.PI*2;
      positions.push(Math.cos(theta)*radiusTop, halfH, Math.sin(theta)*radiusTop);
      normals.push(0,1,0); uvs.push(Math.cos(theta)*0.5+0.5, Math.sin(theta)*0.5+0.5);
    }
    for (let x=0; x<radialSegs; x++) {
      indices.push(topIdx, topIdx+x+2, topIdx+x+1);
    }
    // Bottom cap
    const botIdx = positions.length / 3;
    positions.push(0, -halfH, 0); normals.push(0,-1,0); uvs.push(0.5,0.5);
    for (let x=0; x<=radialSegs; x++) {
      const theta=x/radialSegs*Math.PI*2;
      positions.push(Math.cos(theta)*radiusBottom, -halfH, Math.sin(theta)*radiusBottom);
      normals.push(0,-1,0); uvs.push(Math.cos(theta)*0.5+0.5, Math.sin(theta)*0.5+0.5);
    }
    for (let x=0; x<radialSegs; x++) {
      indices.push(botIdx, botIdx+x+1, botIdx+x+2);
    }
    return {
      positions: new Float32Array(positions),
      normals:   new Float32Array(normals),
      uvs:       new Float32Array(uvs),
      indices:   new Uint16Array(indices)
    };
  },

  createCapsule(radius=0.5, height=1, radialSegs=16, capSegs=8) {
    // Capsule = cylinder body + hemisphere caps
    const positions=[], normals=[], uvs=[], indices=[];
    const halfH = height / 2;
    const stacksPerCap = capSegs;
    // Top hemisphere
    for (let st=0; st<=stacksPerCap; st++) {
      const phi    = (Math.PI / 2) * (st / stacksPerCap);
      const sinPhi = Math.sin(phi), cosPhi = Math.cos(phi);
      for (let sl=0; sl<=radialSegs; sl++) {
        const theta = Math.PI * 2 * sl / radialSegs;
        const x=Math.cos(theta)*sinPhi, y=cosPhi, z=Math.sin(theta)*sinPhi;
        positions.push(x*radius, y*radius+halfH, z*radius);
        normals.push(x, y, z);
        uvs.push(sl/radialSegs, 0.5+st/(stacksPerCap*2));
      }
    }
    // Cylinder body
    for (let y=0; y<=1; y++) {
      const yPos = -halfH + height*y;
      for (let sl=0; sl<=radialSegs; sl++) {
        const theta=Math.PI*2*sl/radialSegs;
        const x=Math.cos(theta), z=Math.sin(theta);
        positions.push(x*radius, yPos, z*radius);
        normals.push(x, 0, z);
        uvs.push(sl/radialSegs, y*0.5);
      }
    }
    // Bottom hemisphere
    for (let st=0; st<=stacksPerCap; st++) {
      const phi    = Math.PI/2 + (Math.PI/2)*(st/stacksPerCap);
      const sinPhi = Math.sin(phi), cosPhi = Math.cos(phi);
      for (let sl=0; sl<=radialSegs; sl++) {
        const theta=Math.PI*2*sl/radialSegs;
        const x=Math.cos(theta)*Math.abs(sinPhi),
              y=cosPhi,
              z=Math.sin(theta)*Math.abs(sinPhi);
        positions.push(x*radius, y*radius-halfH, z*radius);
        normals.push(x*Math.sign(sinPhi), y, z*Math.sign(sinPhi));
        uvs.push(sl/radialSegs, 0.5-st/(stacksPerCap*2));
      }
    }
    const stride = radialSegs + 1;
    const totalStacks = stacksPerCap + 1 + 1 + stacksPerCap;
    for (let st=0; st<totalStacks; st++) {
      for (let sl=0; sl<radialSegs; sl++) {
        const a=st*stride+sl, b=(st+1)*stride+sl;
        indices.push(a,b,a+1, b,b+1,a+1);
      }
    }
    return {
      positions: new Float32Array(positions),
      normals:   new Float32Array(normals),
      uvs:       new Float32Array(uvs),
      indices:   new Uint16Array(indices)
    };
  },

  createGrid(size=50, divisions=50) {
    const step  = size / divisions;
    const half  = size / 2;
    const verts = [];
    for (let i=0; i<=divisions; i++) {
      const p = -half + i * step;
      verts.push(p, 0, -half,  p, 0,  half);
      verts.push(-half, 0, p,   half, 0, p);
    }
    return { positions: new Float32Array(verts) };
  },

  createSkyboxCube() {
    const p = new Float32Array([
      -1, 1,-1, -1,-1,-1,  1,-1,-1,  1,-1,-1,  1, 1,-1, -1, 1,-1,
      -1,-1, 1, -1,-1,-1, -1, 1,-1, -1, 1,-1, -1, 1, 1, -1,-1, 1,
       1,-1,-1,  1,-1, 1,  1, 1, 1,  1, 1, 1,  1, 1,-1,  1,-1,-1,
      -1,-1, 1, -1, 1, 1,  1, 1, 1,  1, 1, 1,  1,-1, 1, -1,-1, 1,
      -1, 1,-1,  1, 1,-1,  1, 1, 1,  1, 1, 1, -1, 1, 1, -1, 1,-1,
      -1,-1,-1, -1,-1, 1,  1,-1,-1,  1,-1,-1, -1,-1, 1,  1,-1, 1
    ]);
    return { positions: p };
  }
};

// ============================================================
//  SECTION 4 — WEBGL RENDERER CLASS
// ============================================================

class UCRenderer {
  constructor(canvas) {
    this.canvas   = canvas;
    this.gl       = null;
    this.programs = {};
    this.meshCache = {};
    this.texCache  = {};

    // Scene data mirrors
    this.objects        = [];
    this.lights         = { ambient:{color:[0.12,0.15,0.2],intensity:1}, directional:[], point:[] };
    this.selectedId     = null;

    // Render state
    this.shadingMode    = 'solid'; // 'solid' | 'wireframe' | 'material'
    this.showGrid       = true;
    this.showGizmos     = true;
    this.showOutline    = true;
    this.skyEnabled     = true;
    this.sky = {
      top:     [0.12, 0.18, 0.30],
      horizon: [0.22, 0.28, 0.38],
      bottom:  [0.10, 0.10, 0.12]
    };

    // Timing
    this.lastTime   = 0;
    this.deltaTime  = 0;
    this.totalTime  = 0;
    this.frameCount = 0;
    this.fps        = 0;
    this._fpsTimer  = 0;
    this._fpsFrames = 0;
    this.animFrameId = null;
    this.running    = false;

    // Temp matrices (reused every frame, no GC pressure)
    this._model   = UCMath.Mat4.create();
    this._view    = UCMath.Mat4.create();
    this._proj    = UCMath.Mat4.create();
    this._mvp     = UCMath.Mat4.create();
    this._normal  = new Float32Array(9);
    this._tmp     = UCMath.Mat4.create();
    this._quat    = UCMath.Quat.create();

    // Skybox / grid GPU buffers
    this._skyBuf  = null;
    this._gridBuf = null;

    this._init();
  }

  // ----------------------------------------------------------
  _init() {
    const gl = this.canvas.getContext('webgl', {
      antialias:              true,
      alpha:                  false,
      depth:                  true,
      stencil:                true,
      powerPreference:        'high-performance',
      preserveDrawingBuffer:  false
    });
    if (!gl) {
      if (window.UCEditor) {
        window.UCEditor.log('error', '❌ WebGL not supported in this browser.', 'Renderer');
      }
      return;
    }
    this.gl = gl;
    // Extensions
    this.extOES  = gl.getExtension('OES_element_index_uint');
    this.extDB   = gl.getExtension('WEBGL_depth_texture');
    this.extAF   = gl.getExtension('EXT_texture_filter_anisotropic');
    // Compile programs
    this.programs.mesh    = this._createProgram(UCShaders.MESH_VERT,    UCShaders.MESH_FRAG);
    this.programs.unlit   = this._createProgram(UCShaders.UNLIT_VERT,   UCShaders.UNLIT_FRAG);
    this.programs.grid    = this._createProgram(UCShaders.GRID_VERT,    UCShaders.GRID_FRAG);
    this.programs.outline = this._createProgram(UCShaders.OUTLINE_VERT, UCShaders.OUTLINE_FRAG);
    this.programs.sky     = this._createProgram(UCShaders.SKY_VERT,     UCShaders.SKY_FRAG);
    // Global GL state
    gl.enable(gl.DEPTH_TEST);
    gl.depthFunc(gl.LEQUAL);
    gl.enable(gl.CULL_FACE);
    gl.cullFace(gl.BACK);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    gl.clearColor(0.11, 0.11, 0.11, 1.0);
    // Upload static GPU buffers
    this._uploadSkybox();
    this._uploadGrid();
    // Pre-build primitive meshes
    this._cachePrimitive('cube',     UCGeometry.createCube());
    this._cachePrimitive('sphere',   UCGeometry.createSphere());
    this._cachePrimitive('plane',    UCGeometry.createPlane());
    this._cachePrimitive('cylinder', UCGeometry.createCylinder());
    this._cachePrimitive('capsule',  UCGeometry.createCapsule());
    if (window.UCEditor) {
      window.UCEditor.log('success', '✅ WebGL context initialized.', 'Renderer');
      window.UCEditor.log('success', `✅ GLSL programs compiled: ${Object.keys(this.programs).join(', ')}`, 'Renderer');
      document.getElementById('status-renderer').textContent = 'WebGL: Active';
    }
    this._resizeObserver();
    this.start();
  }

  // ----------------------------------------------------------
  _createProgram(vertSrc, fragSrc) {
    const gl = this.gl;
    const compile = (type, src) => {
      const sh = gl.createShader(type);
      gl.shaderSource(sh, src);
      gl.compileShader(sh);
      if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
        if (window.UCEditor) window.UCEditor.log('error', '🔴 Shader error: ' + gl.getShaderInfoLog(sh), 'GLSL');
        gl.deleteShader(sh);
        return null;
      }
      return sh;
    };
    const vert = compile(gl.VERTEX_SHADER,   vertSrc);
    const frag = compile(gl.FRAGMENT_SHADER, fragSrc);
    const prog = gl.createProgram();
    gl.attachShader(prog, vert);
    gl.attachShader(prog, frag);
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      if (window.UCEditor) window.UCEditor.log('error', '🔴 Program link error: ' + gl.getProgramInfoLog(prog), 'GLSL');
      return null;
    }
    // Cache uniform / attribute locations
    prog._u = {};
    prog._a = {};
    const uCount = gl.getProgramParameter(prog, gl.ACTIVE_UNIFORMS);
    for (let i=0; i<uCount; i++) {
      const info = gl.getActiveUniform(prog, i);
      prog._u[info.name] = gl.getUniformLocation(prog, info.name);
    }
    const aCount = gl.getProgramParameter(prog, gl.ACTIVE_ATTRIBUTES);
    for (let i=0; i<aCount; i++) {
      const info = gl.getActiveAttrib(prog, i);
      prog._a[info.name] = gl.getAttribLocation(prog, info.name);
    }
    return prog;
  }

  // ----------------------------------------------------------
  _createBuffer(data, target) {
    const gl  = this.gl;
    const buf = gl.createBuffer();
    gl.bindBuffer(target, buf);
    gl.bufferData(target, data, gl.STATIC_DRAW);
    return buf;
  }

  _cachePrimitive(name, geo) {
    const gl = this.gl;
    this.meshCache[name] = {
      positionBuf: this._createBuffer(geo.positions, gl.ARRAY_BUFFER),
      normalBuf:   this._createBuffer(geo.normals,   gl.ARRAY_BUFFER),
      uvBuf:       this._createBuffer(geo.uvs,       gl.ARRAY_BUFFER),
      indexBuf:    this._createBuffer(geo.indices,   gl.ELEMENT_ARRAY_BUFFER),
      indexCount:  geo.indices.length
    };
  }

  _uploadSkybox() {
    const geo    = UCGeometry.createSkyboxCube();
    this._skyBuf = this._createBuffer(geo.positions, this.gl.ARRAY_BUFFER);
  }

  _uploadGrid() {
    const geo     = UCGeometry.createGrid(100, 100);
    this._gridBuf = this._createBuffer(geo.positions, this.gl.ARRAY_BUFFER);
    this._gridVerts = geo.positions.length / 3;
  }

  // ----------------------------------------------------------
  _resizeObserver() {
    const resize = () => {
      const w = this.canvas.clientWidth  * window.devicePixelRatio;
      const h = this.canvas.clientHeight * window.devicePixelRatio;
      if (this.canvas.width !== w || this.canvas.height !== h) {
        this.canvas.width  = w;
        this.canvas.height = h;
        if (this.gl) this.gl.viewport(0, 0, w, h);
      }
    };
    resize();
    window.addEventListener('resize', resize);
    new ResizeObserver(resize).observe(this.canvas);
  }

  // ----------------------------------------------------------
  start() {
    if (this.running) return;
    this.running     = true;
    this.lastTime    = performance.now();
    this.animFrameId = requestAnimationFrame((t) => this._loop(t));
  }

  stop() {
    this.running = false;
    if (this.animFrameId) cancelAnimationFrame(this.animFrameId);
  }

  _loop(timestamp) {
    if (!this.running) return;
    this.deltaTime  = Math.min((timestamp - this.lastTime) / 1000, 0.1);
    this.lastTime   = timestamp;
    this.totalTime += this.deltaTime;
    this.frameCount++;
    // FPS counter
    this._fpsTimer  += this.deltaTime;
    this._fpsFrames += 1;
    if (this._fpsTimer >= 0.5) {
      this.fps = Math.round(this._fpsFrames / this._fpsTimer);
      this._fpsTimer  = 0;
      this._fpsFrames = 0;
      const el = document.getElementById('fps-display');
      if (el) el.textContent = this.fps + ' FPS';
    }
    this._render();
    this.animFrameId = requestAnimationFrame((t) => this._loop(t));
  }

  // ----------------------------------------------------------
  _render() {
    const gl = this.gl;
    if (!gl) return;
    const cam = window.UCCamera;
    if (!cam) return;

    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT | gl.STENCIL_BUFFER_BIT);

    const W = gl.drawingBufferWidth;
    const H = gl.drawingBufferHeight;
    if (W === 0 || H === 0) return;

    gl.viewport(0, 0, W, H);

    cam.updateMatrices(W, H);
    const viewMat = cam.viewMatrix;
    const projMat = cam.projMatrix;
    const camPos  = cam.position;

    // 1. Skybox
    if (this.skyEnabled) this._renderSkybox(viewMat, projMat);

    // 2. Grid
    if (this.showGrid) this._renderGrid(viewMat, projMat, camPos);

    // 3. Scene objects
    const objs = window.UCEditor?.scene?.objects || [];
    this.objects = objs;
    for (const obj of objs) {
      if (!obj.active || !obj.visible) continue;
      if (obj.type.startsWith('light')) continue;
      if (obj.type === 'camera')        continue;
      if (obj.type === 'audio-source')  continue;
      this._renderObject(obj, viewMat, projMat, camPos);
    }

    // 4. Selection outline (stencil-based)
    if (this.showOutline && this.selectedId) {
      const sel = objs.find(o => o.id === this.selectedId);
      if (sel && sel.active && sel.visible && !sel.type.startsWith('light')) {
        this._renderOutline(sel, viewMat, projMat);
      }
    }

    // 5. Gizmos (axes, light icons)
    if (this.showGizmos) {
      this._renderGizmos(objs, viewMat, projMat);
    }

    // Update camera info overlay
    const infoEl = document.getElementById('vp-cam-pos');
    if (infoEl) {
      infoEl.textContent =
        `${camPos[0].toFixed(1)}, ${camPos[1].toFixed(1)}, ${camPos[2].toFixed(1)}`;
    }
  }

  // ----------------------------------------------------------
  _buildModelMatrix(obj) {
    const q   = UCMath.Quat.create();
    const pos = obj.transform.position;
    const rot = obj.transform.rotation;
    const scl = obj.transform.scale;
    UCMath.Quat.fromEuler(q, rot[0], rot[1], rot[2]);
    UCMath.Mat4.fromRotationTranslationScale(
      this._model,
      q,
      pos,
      scl
    );
    return this._model;
  }

  // ----------------------------------------------------------
  _renderObject(obj, viewMat, projMat, camPos) {
    const gl   = this.gl;
    const mesh = this.meshCache[obj.type] || this.meshCache['cube'];
    if (!mesh) return;

    const modelMat = this._buildModelMatrix(obj);

    const prog = this.programs.mesh;
    gl.useProgram(prog);

    // Matrices
    gl.uniformMatrix4fv(prog._u['u_modelMatrix'], false, modelMat);
    gl.uniformMatrix4fv(prog._u['u_viewMatrix'],  false, viewMat);
    gl.uniformMatrix4fv(prog._u['u_projMatrix'],  false, projMat);

    // Normal matrix
    UCMath.Mat4.normalMatrix(this._normal, modelMat);
    gl.uniformMatrix3fv(prog._u['u_normalMatrix'], false, this._normal);

    // Camera position
    gl.uniform3fv(prog._u['u_cameraPos'], camPos);

    // Material
    const mat = obj.material || {};
    gl.uniform3fv(prog._u['u_baseColor'],  mat.color     || [0.6, 0.6, 0.65]);
    gl.uniform1f(prog._u['u_metallic'],    mat.metallic  !== undefined ? mat.metallic  : 0.0);
    gl.uniform1f(prog._u['u_roughness'],   mat.roughness !== undefined ? mat.roughness : 0.7);
    gl.uniform1f(prog._u['u_emission'],    mat.emission  !== undefined ? mat.emission  : 0.0);
    gl.uniform1i(prog._u['u_useTexture'],  0);

    // Lighting
    const L = this.lights;
    gl.uniform3fv(prog._u['u_ambientColor'],     L.ambient.color);
    gl.uniform1f(prog._u['u_ambientIntensity'],  L.ambient.intensity);

    // Find first directional light from scene
    const sceneObjs = window.UCEditor?.scene?.objects || [];
    const dirObj    = sceneObjs.find(o => o.type === 'light-dir');
    if (dirObj) {
      const r = dirObj.transform.rotation;
      const q = UCMath.Quat.create();
      UCMath.Quat.fromEuler(q, r[0], r[1], r[2]);
      const down = [0, -1, 0];
      const ld   = UCMath.Vec3.create();
      UCMath.Vec3.set(ld, down[0], down[1], down[2]);
      gl.uniform3fv(prog._u['u_dirLightDir'],       ld);
      gl.uniform3fv(prog._u['u_dirLightColor'],     dirObj.material?.color || [1,1,1]);
      gl.uniform1f(prog._u['u_dirLightIntensity'],  dirObj.material?.intensity || 1.0);
    } else {
      gl.uniform3fv(prog._u['u_dirLightDir'],      [0.4, -0.8, 0.3]);
      gl.uniform3fv(prog._u['u_dirLightColor'],    [1.0, 0.98, 0.92]);
      gl.uniform1f(prog._u['u_dirLightIntensity'], 1.2);
    }

    // First point light
    const ptObj = sceneObjs.find(o => o.type === 'light-point');
    if (ptObj) {
      gl.uniform3fv(prog._u['u_pointLight0Pos'],       ptObj.transform.position);
      gl.uniform3fv(prog._u['u_pointLight0Color'],     ptObj.material?.color || [1,1,1]);
      gl.uniform1f(prog._u['u_pointLight0Intensity'],  ptObj.material?.intensity || 2.0);
      gl.uniform1f(prog._u['u_pointLight0Range'],      ptObj.material?.range || 10.0);
    } else {
      gl.uniform3fv(prog._u['u_pointLight0Pos'],      [0, 5, 0]);
      gl.uniform3fv(prog._u['u_pointLight0Color'],    [0,0,0]);
      gl.uniform1f(prog._u['u_pointLight0Intensity'], 0);
      gl.uniform1f(prog._u['u_pointLight0Range'],     1);
    }

    // Attributes
    this._bindMeshBuffers(prog, mesh);

    // Wireframe override
    if (this.shadingMode === 'wireframe') {
      gl.uniform3fv(prog._u['u_baseColor'], [0.3, 0.8, 1.0]);
      gl.uniform1f(prog._u['u_emission'],   1.0);
    }

    gl.drawElements(gl.TRIANGLES, mesh.indexCount, gl.UNSIGNED_SHORT, 0);
    this._unbindMeshBuffers(prog);
  }

  // ----------------------------------------------------------
  _bindMeshBuffers(prog, mesh) {
    const gl = this.gl;
    if (prog._a['a_position'] !== undefined && prog._a['a_position'] >= 0) {
      gl.bindBuffer(gl.ARRAY_BUFFER, mesh.positionBuf);
      gl.enableVertexAttribArray(prog._a['a_position']);
      gl.vertexAttribPointer(prog._a['a_position'], 3, gl.FLOAT, false, 0, 0);
    }
    if (prog._a['a_normal'] !== undefined && prog._a['a_normal'] >= 0) {
      gl.bindBuffer(gl.ARRAY_BUFFER, mesh.normalBuf);
      gl.enableVertexAttribArray(prog._a['a_normal']);
      gl.vertexAttribPointer(prog._a['a_normal'], 3, gl.FLOAT, false, 0, 0);
    }
    if (prog._a['a_uv'] !== undefined && prog._a['a_uv'] >= 0) {
      gl.bindBuffer(gl.ARRAY_BUFFER, mesh.uvBuf);
      gl.enableVertexAttribArray(prog._a['a_uv']);
      gl.vertexAttribPointer(prog._a['a_uv'], 2, gl.FLOAT, false, 0, 0);
    }
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, mesh.indexBuf);
  }

  _unbindMeshBuffers(prog) {
    const gl = this.gl;
    if (prog._a['a_position'] >= 0) gl.disableVertexAttribArray(prog._a['a_position']);
    if (prog._a['a_normal']   >= 0) gl.disableVertexAttribArray(prog._a['a_normal']);
    if (prog._a['a_uv']       >= 0) gl.disableVertexAttribArray(prog._a['a_uv']);
  }

  // ----------------------------------------------------------
  _renderOutline(obj, viewMat, projMat) {
    const gl   = this.gl;
    const mesh = this.meshCache[obj.type] || this.meshCache['cube'];
    if (!mesh) return;
    const modelMat = this._buildModelMatrix(obj);
    const prog = this.programs.outline;
    gl.useProgram(prog);
    gl.cullFace(gl.FRONT);
    gl.uniformMatrix4fv(prog._u['u_modelMatrix'],   false, modelMat);
    gl.uniformMatrix4fv(prog._u['u_viewMatrix'],    false, viewMat);
    gl.uniformMatrix4fv(prog._u['u_projMatrix'],    false, projMat);
    gl.uniform1f(prog._u['u_outlineWidth'],  0.025);
    gl.uniform3fv(prog._u['u_outlineColor'], [0.2, 0.6, 1.0]);
    if (prog._a['a_position'] >= 0) {
      gl.bindBuffer(gl.ARRAY_BUFFER, mesh.positionBuf);
      gl.enableVertexAttribArray(prog._a['a_position']);
      gl.vertexAttribPointer(prog._a['a_position'], 3, gl.FLOAT, false, 0, 0);
    }
    if (prog._a['a_normal'] >= 0) {
      gl.bindBuffer(gl.ARRAY_BUFFER, mesh.normalBuf);
      gl.enableVertexAttribArray(prog._a['a_normal']);
      gl.vertexAttribPointer(prog._a['a_normal'], 3, gl.FLOAT, false, 0, 0);
    }
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, mesh.indexBuf);
    gl.drawElements(gl.TRIANGLES, mesh.indexCount, gl.UNSIGNED_SHORT, 0);
    if (prog._a['a_position'] >= 0) gl.disableVertexAttribArray(prog._a['a_position']);
    if (prog._a['a_normal']   >= 0) gl.disableVertexAttribArray(prog._a['a_normal']);
    gl.cullFace(gl.BACK);
  }

  // ----------------------------------------------------------
  _renderSkybox(viewMat, projMat) {
    const gl   = this.gl;
    const prog = this.programs.sky;
    gl.useProgram(prog);
    gl.depthMask(false);
    gl.uniformMatrix4fv(prog._u['u_viewMatrix'], false, viewMat);
    gl.uniformMatrix4fv(prog._u['u_projMatrix'], false, projMat);
    gl.uniform3fv(prog._u['u_skyTop'],     this.sky.top);
    gl.uniform3fv(prog._u['u_skyHorizon'], this.sky.horizon);
    gl.uniform3fv(prog._u['u_skyBottom'],  this.sky.bottom);
    gl.bindBuffer(gl.ARRAY_BUFFER, this._skyBuf);
    const aPos = prog._a['a_position'];
    if (aPos >= 0) {
      gl.enableVertexAttribArray(aPos);
      gl.vertexAttribPointer(aPos, 3, gl.FLOAT, false, 0, 0);
    }
    gl.disable(gl.CULL_FACE);
    gl.drawArrays(gl.TRIANGLES, 0, 36);
    gl.enable(gl.CULL_FACE);
    gl.depthMask(true);
    if (aPos >= 0) gl.disableVertexAttribArray(aPos);
  }

  // ----------------------------------------------------------
  _renderGrid(viewMat, projMat, camPos) {
    const gl   = this.gl;
    const prog = this.programs.grid;
    gl.useProgram(prog);
    gl.uniformMatrix4fv(prog._u['u_viewMatrix'],   false, viewMat);
    gl.uniformMatrix4fv(prog._u['u_projMatrix'],   false, projMat);
    gl.uniform3fv(prog._u['u_cameraPos'],  camPos);
    gl.uniform1f(prog._u['u_gridSize'],    1.0);
    gl.uniform1f(prog._u['u_lineWidth'],   1.5);
    gl.uniform3fv(prog._u['u_gridColor'],  [0.35, 0.35, 0.4]);
    gl.uniform1f(prog._u['u_fadeDistance'], 60.0);
    gl.bindBuffer(gl.ARRAY_BUFFER, this._gridBuf);
    const aPos = prog._a['a_position'];
    if (aPos >= 0) {
      gl.enableVertexAttribArray(aPos);
      gl.vertexAttribPointer(aPos, 3, gl.FLOAT, false, 0, 0);
    }
    gl.disable(gl.CULL_FACE);
    gl.drawArrays(gl.LINES, 0, this._gridVerts);
    gl.enable(gl.CULL_FACE);
    if (aPos >= 0) gl.disableVertexAttribArray(aPos);
  }

  // ----------------------------------------------------------
  _renderGizmos(objs, viewMat, projMat) {
    const gl   = this.gl;
    const prog = this.programs.unlit;
    gl.useProgram(prog);
    gl.uniformMatrix4fv(prog._u['u_viewMatrix'], false, viewMat);
    gl.uniformMatrix4fv(prog._u['u_projMatrix'], false, projMat);
    gl.disable(gl.DEPTH_TEST);
    // Render axes for selected object
    if (this.selectedId) {
      const sel = objs.find(o => o.id === this.selectedId);
      if (sel && this.currentTool === 'move') {
        this._renderMoveGizmo(sel, viewMat, projMat);
      }
    }
    // Light icons
    for (const obj of objs) {
      if (obj.type.startsWith('light') || obj.type === 'camera') {
        this._renderLightIcon(obj, viewMat, projMat);
      }
    }
    gl.enable(gl.DEPTH_TEST);
  }

  // ----------------------------------------------------------
  _renderMoveGizmo(obj, viewMat, projMat) {
    // Render 3 axis lines at object position
    const gl   = this.gl;
    const prog = this.programs.unlit;
    const pos  = obj.transform.position;
    const len  = 1.0;
    const axes = [
      { color:[1,0.15,0.15], dir:[len,0,0] },
      { color:[0.15,1,0.15], dir:[0,len,0] },
      { color:[0.15,0.15,1], dir:[0,0,len] }
    ];
    UCMath.Mat4.identity(this._model);
    UCMath.Mat4.translate(this._model, this._model, pos);
    gl.uniformMatrix4fv(prog._u['u_modelMatrix'], false, this._model);
    for (const axis of axes) {
      const verts = new Float32Array([0,0,0, ...axis.dir]);
      const buf   = this.gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, buf);
      gl.bufferData(gl.ARRAY_BUFFER, verts, gl.DYNAMIC_DRAW);
      const aPos = prog._a['a_position'];
      if (aPos >= 0) {
        gl.enableVertexAttribArray(aPos);
        gl.vertexAttribPointer(aPos, 3, gl.FLOAT, false, 0, 0);
      }
      gl.uniform3fv(prog._u['u_color'], axis.color);
      gl.uniform1f(prog._u['u_alpha'],  1.0);
      gl.lineWidth(2.0);
      gl.drawArrays(gl.LINES, 0, 2);
      if (aPos >= 0) gl.disableVertexAttribArray(aPos);
      gl.deleteBuffer(buf);
    }
  }

  // ----------------------------------------------------------
  _renderLightIcon(obj, viewMat, projMat) {
    const gl   = this.gl;
    const prog = this.programs.unlit;
    const pos  = obj.transform.position;
    const colors = {
      'light-dir':   [1.0, 0.95, 0.5],
      'light-point': [1.0, 0.8,  0.3],
      'light-spot':  [0.8, 0.9,  1.0],
      'camera':      [0.5, 0.8,  1.0]
    };
    const color = colors[obj.type] || [1,1,1];
    const size  = 0.12;
    const pts   = [];
    const segs  = 12;
    for (let i=0; i<segs; i++) {
      const a0 = (i/segs)     * Math.PI * 2;
      const a1 = ((i+1)/segs) * Math.PI * 2;
      pts.push(
        pos[0]+Math.cos(a0)*size, pos[1]+Math.sin(a0)*size, pos[2],
        pos[0]+Math.cos(a1)*size, pos[1]+Math.sin(a1)*size, pos[2]
      );
    }
    const buf  = this.gl.createBuffer();
    UCMath.Mat4.identity(this._model);
    gl.uniformMatrix4fv(prog._u['u_modelMatrix'], false, this._model);
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(pts), gl.DYNAMIC_DRAW);
    const aPos = prog._a['a_position'];
    if (aPos >= 0) {
      gl.enableVertexAttribArray(aPos);
      gl.vertexAttribPointer(aPos, 3, gl.FLOAT, false, 0, 0);
    }
    gl.uniform3fv(prog._u['u_color'], color);
    gl.uniform1f(prog._u['u_alpha'],  0.9);
    gl.drawArrays(gl.LINES, 0, segs * 2);
    if (aPos >= 0) gl.disableVertexAttribArray(aPos);
    gl.deleteBuffer(buf);
  }

  // ----------------------------------------------------------
  setSelectedId(id)       { this.selectedId  = id;   }
  setShadingMode(mode)    { this.shadingMode  = mode; }
  setShowGrid(v)          { this.showGrid     = v;    }
  setShowGizmos(v)        { this.showGizmos   = v;    }
  setCurrentTool(t)       { this.currentTool  = t;    }

  getSkySettings()        { return this.sky; }
  setSkyColor(t, color)   { this.sky[t] = color; }
}

// ============================================================
//  SECTION 5 — ORBIT CAMERA
// ============================================================

class UCOrbitCamera {
  constructor() {
    this.position   = new Float32Array([0, 5, 12]);
    this.target     = new Float32Array([0, 0,  0]);
    this.up         = new Float32Array([0, 1,  0]);

    // Spherical coordinates
    this.phi        = 0.5;   // vertical angle (radians)
    this.theta      = 0.0;   // horizontal angle (radians)
    this.radius     = 13.0;  // distance from target

    // Limits
    this.minPhi     = 0.05;
    this.maxPhi     = Math.PI - 0.05;
    this.minRadius  = 0.5;
    this.maxRadius  = 500;

    this.fov        = 60 * UCMath.DEG2RAD;
    this.near       = 0.01;
    this.far        = 1000;
    this.ortho      = false;
    this.orthoSize  = 10;

    this.viewMatrix = UCMath.Mat4.create();
    this.projMatrix = UCMath.Mat4.create();

    // Pan speed
    this.panSpeed   = 0.005;
    this.orbitSpeed = 0.006;
    this.zoomSpeed  = 1.1;

    // Mouse state
    this._mouseDown  = false;
    this._rmb        = false;
    this._mmb        = false;
    this._lastX      = 0;
    this._lastY      = 0;

    this._bindInput();
    this._updatePosition();
  }

  // ----------------------------------------------------------
  _bindInput() {
    const canvas = document.getElementById('viewport-canvas');
    if (!canvas) return;

    canvas.addEventListener('mousedown', (e) => {
      this._mouseDown = true;
      this._rmb = e.button === 2;
      this._mmb = e.button === 1;
      this._lastX = e.clientX;
      this._lastY = e.clientY;
      e.preventDefault();
    });

    window.addEventListener('mouseup', () => {
      this._mouseDown = false;
      this._rmb = false;
      this._mmb = false;
    });

    window.addEventListener('mousemove', (e) => {
      if (!this._mouseDown) return;
      const dx = e.clientX - this._lastX;
      const dy = e.clientY - this._lastY;
      this._lastX = e.clientX;
      this._lastY = e.clientY;

      if (this._rmb) {
        // Orbit
        this.theta -= dx * this.orbitSpeed;
        this.phi   -= dy * this.orbitSpeed;
        this.phi    = Math.max(this.minPhi, Math.min(this.maxPhi, this.phi));
        this._updatePosition();
      } else if (this._mmb) {
        // Pan
        this._pan(dx, dy);
      }
    });

    canvas.addEventListener('wheel', (e) => {
      e.preventDefault();
      const factor = e.deltaY > 0 ? this.zoomSpeed : 1 / this.zoomSpeed;
      this.radius  = Math.max(this.minRadius, Math.min(this.maxRadius, this.radius * factor));
      this._updatePosition();
    }, { passive: false });

    canvas.addEventListener('contextmenu', e => e.preventDefault());
  }

  // ----------------------------------------------------------
  _pan(dx, dy) {
    const V3 = UCMath.Vec3;
    // Right vector = cross(forward, up)
    const forward = V3.create();
    V3.sub(forward, this.target, this.position);
    V3.normalize(forward, forward);
    const right = V3.create();
    V3.cross(right, forward, this.up);
    V3.normalize(right, right);
    const upVec = V3.create();
    V3.cross(upVec, right, forward);
    V3.normalize(upVec, upVec);

    const panScale = this.radius * this.panSpeed;
    const dr = V3.scale(V3.create(), right,  -dx * panScale);
    const du = V3.scale(V3.create(), upVec,   dy * panScale);
    const delta = V3.add(V3.create(), dr, du);

    V3.add(this.target,   this.target,   delta);
    V3.add(this.position, this.position, delta);
  }

  // ----------------------------------------------------------
  _updatePosition() {
    const sin  = Math.sin;
    const cos  = Math.cos;
    this.position[0] = this.target[0] + this.radius * sin(this.phi) * sin(this.theta);
    this.position[1] = this.target[1] + this.radius * cos(this.phi);
    this.position[2] = this.target[2] + this.radius * sin(this.phi) * cos(this.theta);
  }

  // ----------------------------------------------------------
  updateMatrices(width, height) {
    UCMath.Mat4.lookAt(this.viewMatrix, this.position, this.target, this.up);
    const aspect = width / height;
    if (this.ortho) {
      const hs = this.orthoSize * 0.5;
      UCMath.Mat4.ortho(this.projMatrix, -hs * aspect, hs * aspect, -hs, hs, this.near, this.far);
    } else {
      UCMath.Mat4.perspective(this.projMatrix, this.fov, aspect, this.near, this.far);
    }
  }

  // ----------------------------------------------------------
  focusOn(position, radius = 3) {
    UCMath.Vec3.copy(this.target, position);
    this.radius = radius + 3;
    this._updatePosition();
  }

  frameAll(objects) {
    if (!objects.length) return;
    let cx=0, cy=0, cz=0;
    for (const obj of objects) {
      cx += obj.transform.position[0];
      cy += obj.transform.position[1];
      cz += obj.transform.position[2];
    }
    this.target[0] = cx / objects.length;
    this.target[1] = cy / objects.length;
    this.target[2] = cz / objects.length;
    this.radius = Math.max(8, objects.length * 2);
    this._updatePosition();
  }

  toggleOrtho() {
    this.ortho = !this.ortho;
    const btn = document.getElementById('vp-ortho-toggle');
    if (btn) btn.textContent = this.ortho ? 'Ortho' : 'Persp';
    return this.ortho;
  }

  setFOV(deg) {
    this.fov = deg * UCMath.DEG2RAD;
  }
}

// ============================================================
//  SECTION 6 — RAYCASTING (Mouse Picking)
// ============================================================

class UCRaycaster {
  constructor() {
    this._ray = {
      origin:    UCMath.Vec3.create(),
      direction: UCMath.Vec3.create()
    };
  }

  // Build a ray from NDC screen coords
  fromCamera(ndcX, ndcY, camera) {
    const invProj = UCMath.Mat4.create();
    const invView = UCMath.Mat4.create();
    UCMath.Mat4.invert(invProj, camera.projMatrix);
    UCMath.Mat4.invert(invView, camera.viewMatrix);

    // Unproject near/far
    const nearPt = new Float32Array([ndcX, ndcY, -1, 1]);
    const farPt  = new Float32Array([ndcX, ndcY,  1, 1]);
    UCMath.Vec4.transformMat4(nearPt, nearPt, invProj);
    UCMath.Vec4.transformMat4(farPt,  farPt,  invProj);
    const nearW = nearPt[3] || 1;
    const farW  = farPt[3]  || 1;
    nearPt[0]/=nearW; nearPt[1]/=nearW; nearPt[2]/=nearW;
    farPt[0] /=farW;  farPt[1] /=farW;  farPt[2] /=farW;

    const near4 = new Float32Array([nearPt[0],nearPt[1],nearPt[2],1]);
    const far4  = new Float32Array([farPt[0], farPt[1], farPt[2], 1]);
    UCMath.Vec4.transformMat4(near4, near4, invView);
    UCMath.Vec4.transformMat4(far4,  far4,  invView);
    const nw = near4[3]||1, fw = far4[3]||1;

    this._ray.origin[0] = near4[0]/nw;
    this._ray.origin[1] = near4[1]/nw;
    this._ray.origin[2] = near4[2]/nw;

    const dir = UCMath.Vec3.create();
    dir[0] = far4[0]/fw - this._ray.origin[0];
    dir[1] = far4[1]/fw - this._ray.origin[1];
    dir[2] = far4[2]/fw - this._ray.origin[2];
    UCMath.Vec3.normalize(this._ray.direction, dir);
    return this._ray;
  }

  // AABB intersection test
  intersectAABB(ray, center, halfSize) {
    const V3  = UCMath.Vec3;
    const inv = V3.create();
    inv[0] = 1/ray.direction[0];
    inv[1] = 1/ray.direction[1];
    inv[2] = 1/ray.direction[2];
    const t1 = (center[0]-halfSize[0]-ray.origin[0])*inv[0];
    const t2 = (center[0]+halfSize[0]-ray.origin[0])*inv[0];
    const t3 = (center[1]-halfSize[1]-ray.origin[1])*inv[1];
    const t4 = (center[1]+halfSize[1]-ray.origin[1])*inv[1];
    const t5 = (center[2]-halfSize[2]-ray.origin[2])*inv[2];
    const t6 = (center[2]+halfSize[2]-ray.origin[2])*inv[2];
    const tMin = Math.max(Math.max(Math.min(t1,t2),Math.min(t3,t4)),Math.min(t5,t6));
    const tMax = Math.min(Math.min(Math.max(t1,t2),Math.max(t3,t4)),Math.max(t5,t6));
    if (tMax < 0 || tMin > tMax) return null;
    return tMin < 0 ? tMax : tMin;
  }

  // Test ray against all scene objects
  intersectScene(ray, objects) {
    let closest = null;
    let closestDist = Infinity;
    for (const obj of objects) {
      if (!obj.active || !obj.visible) continue;
      const pos  = obj.transform.position;
      const scl  = obj.transform.scale;
      const half = [scl[0]*0.5, scl[1]*0.5, scl[2]*0.5];
      const dist = this.intersectAABB(ray, pos, half);
      if (dist !== null && dist < closestDist) {
        closestDist = dist;
        closest     = obj;
      }
    }
    return closest;
  }
}

// ============================================================
//  SECTION 7 — AXIS GIZMO (Corner Widget)
// ============================================================

class UCAxisGizmo {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx    = canvas.getContext('2d');
    this.size   = 64;
    canvas.width  = this.size;
    canvas.height = this.size;
  }

  draw(camera) {
    const ctx    = this.ctx;
    const size   = this.size;
    const center = size / 2;
    const armLen = 22;
    ctx.clearRect(0, 0, size, size);

    // Background circle
    ctx.beginPath();
    ctx.arc(center, center, center - 2, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(20,20,25,0.7)';
    ctx.fill();
    ctx.strokeStyle = 'rgba(255,255,255,0.1)';
    ctx.lineWidth = 1;
    ctx.stroke();

    // Extract axes from view matrix
    const vm = camera.viewMatrix;
    const axes = [
      { dir:[ vm[0],  vm[4],  vm[8]],  color:'#f44',  neg:'#722', label:'X' },
      { dir:[ vm[1],  vm[5],  vm[9]],  color:'#4f4',  neg:'#272', label:'Y' },
      { dir:[ vm[2],  vm[6],  vm[10]], color:'#44f',  neg:'#227', label:'Z' }
    ];

    // Sort by depth (draw back-to-front)
    axes.sort((a, b) => a.dir[2] - b.dir[2]);

    for (const axis of axes) {
      const ex = center + axis.dir[0] * armLen;
      const ey = center - axis.dir[1] * armLen;
      ctx.beginPath();
      ctx.moveTo(center, center);
      ctx.lineTo(ex, ey);
      ctx.strokeStyle = axis.color;
      ctx.lineWidth   = 2;
      ctx.stroke();
      // Dot
      ctx.beginPath();
      ctx.arc(ex, ey, 5, 0, Math.PI * 2);
      ctx.fillStyle = axis.color;
      ctx.fill();
      // Label
      ctx.fillStyle   = '#fff';
      ctx.font        = 'bold 8px Segoe UI';
      ctx.textAlign   = 'center';
      ctx.textBaseline= 'middle';
      ctx.fillText(axis.label, ex, ey);
    }
  }
}

// ============================================================
//  SECTION 8 — VIEWPORT CONTROLLER
//  Wires renderer + camera + raycaster + gizmo widget together
// ============================================================

class UCViewportController {
  constructor() {
    this.canvas      = document.getElementById('viewport-canvas');
    this.gizmoCanvas = document.getElementById('gizmo-canvas');
    if (!this.canvas) return;

    this.renderer  = new UCRenderer(this.canvas);
    this.camera    = new UCOrbitCamera();
    this.raycaster = new UCRaycaster();
    this.axisGizmo = new UCAxisGizmo(this.gizmoCanvas);

    // Expose camera globally for renderer
    window.UCCamera = this.camera;

    this._bindViewportClicks();
    this._bindViewportToolbar();
    this._gizmoLoop();

    // Hook editor selection changes into renderer
    this._hookEditorSelection();

    if (window.UCEditor) {
      window.UCEditor.log('success', '✅ Viewport controller ready.', 'Viewport');
      window.UCEditor.log('info',    '🖱️  RMB+drag: orbit  |  MMB+drag: pan  |  scroll: zoom', 'Viewport');
    }
  }

  // ----------------------------------------------------------
  _bindViewportClicks() {
    this.canvas.addEventListener('click', (e) => {
      if (!window.UCEditor || !window.UCCamera) return;
      const rect = this.canvas.getBoundingClientRect();
      const ndcX = ((e.clientX - rect.left) / rect.width)  * 2 - 1;
      const ndcY = ((e.clientY - rect.top)  / rect.height) * -2 + 1;
      const ray  = this.raycaster.fromCamera(ndcX, ndcY, window.UCCamera);
      const objs = window.UCEditor.scene.objects || [];
      const hit  = this.raycaster.intersectScene(ray, objs);
      if (hit) {
        window.UCEditor.selectObject(hit.id);
        this.renderer.setSelectedId(hit.id);
      } else {
        window.UCEditor.deselectAll();
        this.renderer.setSelectedId(null);
      }
    });
  }

  // ----------------------------------------------------------
  _bindViewportToolbar() {
    document.getElementById('vp-shading-solid')?.addEventListener('click', () => {
      this.renderer.setShadingMode('solid');
    });
    document.getElementById('vp-shading-wire')?.addEventListener('click', () => {
      this.renderer.setShadingMode('wireframe');
    });
    document.getElementById('vp-shading-material')?.addEventListener('click', () => {
      this.renderer.setShadingMode('material');
    });
    document.getElementById('vp-toggle-grid')?.addEventListener('click', () => {
      this.renderer.setShowGrid(!this.renderer.showGrid);
    });
    document.getElementById('vp-toggle-gizmos')?.addEventListener('click', () => {
      this.renderer.setShowGizmos(!this.renderer.showGizmos);
    });
    document.getElementById('vp-ortho-toggle')?.addEventListener('click', () => {
      const isOrtho = this.camera.toggleOrtho();
      document.getElementById('viewport-info').firstChild.textContent =
        isOrtho ? 'Orthographic' : 'Perspective';
    });
  }

  // ----------------------------------------------------------
  _hookEditorSelection() {
    const orig = window.UCEditor?.selectObject?.bind(window.UCEditor);
    if (!orig) return;
    window.UCEditor.selectObject = (id) => {
      orig(id);
      if (this.renderer) this.renderer.setSelectedId(id);
      if (this.renderer) this.renderer.setCurrentTool(window.UCEditor.currentTool);
    };
    const origDesel = window.UCEditor?.deselectAll?.bind(window.UCEditor);
    if (origDesel) {
      window.UCEditor.deselectAll = () => {
        origDesel();
        if (this.renderer) this.renderer.setSelectedId(null);
      };
    }
    // Hook tool change
    ['move','rotate','scale','rect'].forEach(tool => {
      const btn = document.getElementById('tool-' + tool);
      if (btn) {
        btn.addEventListener('click', () => {
          if (this.renderer) this.renderer.setCurrentTool(tool);
        });
      }
    });
    // Hook frame-all / frame-selected
    const origAction = window.UCEditor?.handleAction?.bind(window.UCEditor);
    if (origAction) {
      window.UCEditor.handleAction = (action) => {
        if (action === 'frame-all') {
          const objs = window.UCEditor?.scene?.objects || [];
          if (objs.length) this.camera.frameAll(objs);
        } else if (action === 'frame-selected' && window.UCEditor.selectedObject) {
          const obj = window.UCEditor.selectedObject;
          this.camera.focusOn(obj.transform.position, 3);
        }
        origAction(action);
      };
    }
  }

  // ----------------------------------------------------------
  _gizmoLoop() {
    const draw = () => {
      if (this.axisGizmo && window.UCCamera) {
        this.axisGizmo.draw(window.UCCamera);
      }
      requestAnimationFrame(draw);
    };
    draw();
  }
}

// ============================================================
//  SECTION 9 — BOOT
// ============================================================

// Wait until the editor shell is fully initialized before
// creating the renderer, so the canvas element exists.
window.addEventListener('load', () => {
  // Give the editor init a moment to run
  const tryBoot = (attempts = 0) => {
    const canvas = document.getElementById('viewport-canvas');
    if (!canvas) {
      if (attempts < 20) setTimeout(() => tryBoot(attempts + 1), 150);
      return;
    }
    // Boot the viewport controller — this creates the renderer + camera
    window.UCViewport = new UCViewportController();
    if (window.UCEditor) {
      window.UCEditor.log('engine', '🚀 Renderer + Camera + Viewport booted.', 'Engine');
    }
  };
  setTimeout(() => tryBoot(), 2200);
});
