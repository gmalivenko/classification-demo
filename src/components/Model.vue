<template>
  <div class="content">
    <div class="spinner-border" role="status" v-if="!model">
      <span class="sr-only">Loading...</span>
    </div>
    <div class="inference-window" v-else>
      <div class="input-window">
        <div class="model-info">
          <p> Model: {{model.full_name}} </p>
          <p> Weights source: {{model.weights_source}} </p>
          <p> Reported top1 error: {{model.reported_top1_error}} </p>
          <p> Reported top5 error: {{model.reported_top5_error}} </p>
          <p> FLOPS: {{model.flops}} </p>
          <p> Parameters: {{model.num_params}} </p>
          <p> Paper: <a v-bind="{'href': model.paper}"> [link] </a> </p>
          <p> {{model.description}} </p>
          <hr />
          <div class="spinner-border" role="status" v-if="evaluating">
            <span class="sr-only">Loading...</span>
          </div>
          <div class="pointer" @click="evaluate" v-else> 
            [START] 
          </div>
          <div v-if="results" class="results">
            Predicted labels:
            <hr />
            <p v-for="result in results" v-bind:key="result.className"> >> {{result.className}} : {{result.probability | percentage}} </p>

            <hr />
            Elapsed time: {{elapsed | round}} ms
          </div>
        </div>

        <div class="file-upload" @click="trigger">
          <div class="image-preview" v-if="imageData.length > 0">
              <img ref="imageInput" class="preview" :src="imageData" :style="{height: imgSize + 'px',  width: imgSize + 'px' }">
          </div>
          <div class="file-upload-form">
              <a > Click to change image </a>
              <input ref="fileInput" class="input-file" type="file" @change="previewImage" accept="image/*">
          </div>
        </div>

      </div>
    </div>
   
  </div>
</template>

<script>
import models from '../models.js'
import * as tf from '@tensorflow/tfjs';
import imagenet1k from '../imagenet_classes.js'

// console.log(imagenet1k)

const TOP_K_PREDICTIONS = 5

// Fast check Tensorflow.js:
let d = tf.tensor3d([[[3.0]]]);
let d_squared = d.square();
d_squared.data().then((d_squared) => {
  if (d_squared[0] != 9) {
    tf.setBackend('cpu');
    console.log('GPU error detected. Switch to CPU batched.')
  }
  else {
    console.log('Using WebGL backend')
  }
})

// Predict with specific model
async function predict(params, image) {
  const model = await tf.loadGraphModel(params.weights_src[0]);

  const startTime = performance.now();

  // tf.fromPixels() returns a Tensor from an image element.
  const img =  tf.browser.fromPixels(image).toFloat();

  // Normalize the image
  const normalized = img.sub(params.mean).div(params.std);
  console.log(await normalized.data())

  // Reshape to a single-element batch so we can pass it to predict.
  const batched = tf.transpose(normalized, [2, 0, 1]).reshape([1, 3, params.window_size, params.window_size]);

  // Make a prediction through mobilenet.
  const logits = await (model.executeAsync(batched))
  console.log(await logits.data())

  const classes = await getTopKClasses(logits.softmax(), TOP_K_PREDICTIONS, params.classes);

  // Convert logits to probabilities and class names.
  
  const totalTime = performance.now() - startTime;

  await logits.dispose();
  await batched.dispose();
  await normalized.dispose();
  await img.dispose();

  await model.dispose();

  // Clean up memory
  await tf.disposeVariables();

  return {classes, totalTime};
}

// Return top-k classes
async function getTopKClasses(logits, topK, dataset) {
  const values = await logits.data();
  
  let classes = {}

  if (dataset == 'imagenet-1000') {
    classes = imagenet1k;
  } else {
    console.log('Unknown class set')
  }
  
  const valuesAndIndices = [];
  for (let i = 0; i < values.length; i++) {
    valuesAndIndices.push({value: values[i], index: i});
  }

  valuesAndIndices.sort((a, b) => {
    return b.value - a.value;
  });
  
  const topkValues = new Float32Array(topK);
  const topkIndices = new Int32Array(topK);
  
  for (let i = 0; i < topK; i++) {
    topkValues[i] = valuesAndIndices[i].value;
    topkIndices[i] = valuesAndIndices[i].index;
  }

  const topClassesAndProbs = [];
  for (let i = 0; i < topkIndices.length; i++) {
    topClassesAndProbs.push({
      className: classes[topkIndices[i]],
      probability: topkValues[i]
    })
  }

  return topClassesAndProbs;
}


export default {
  name: 'HelloWorld',
  props: {
    msg: String,
    dataImages: String,
  },
  methods: {
    async evaluate () {
      console.log('Run model inference')
      if (!this.imageData) {
        console.log('You have to choose image first')
      }
      this.evaluating = true

      let result = await predict(this.model, this.$refs.imageInput)
      this.results = result.classes
      this.elapsed = result.totalTime

      console.log(result)
      this.evaluating = false
    },
    trigger () {
     this.$refs.fileInput.click()
    },
    previewImage: function(event) {
      // Reference to the DOM input element
      var input = event.target;
      // Ensure that you have a file before attempting to read it
      if (input.files && input.files[0]) {
        // create a new FileReader to read this image and convert to base64 format
        var reader = new FileReader();
        // Define a callback function to run, when FileReader finishes its job
        reader.onload = (e) => {
          // Note: arrow function used here, so that "this.imageData" refers to the imageData of Vue component
          // Read image as base64 and set to imageData
          this.imageData = e.target.result;
        }
        // Start the reader job - read file as a data url (base64 format)
        reader.readAsDataURL(input.files[0]);
      }
    }
  },
  data () {
    console.log(this.$route.params.model)
    
    let model = models.models.find((e) => {
      return e.name == this.$route.params.model
    })

    console.log(model)

    return {
      imageData: require('../assets/cat.jpg'),
      model: model,
      evaluating: false,
      results: false,
      elapsed: false,
    }
  },
  computed: {
    imgSize() {
        return this.model.window_size;
    }
  }
}
</script>

<!-- Add "scoped" attribute to limit CSS to this component only -->
<style scoped>
a {
  text-decoration: none;
  color: #42b983;
}

.pointer {
  cursor: pointer;
  border: #2BA84A solid 2px;
  height: 50px;
  display: flex;
  align-items: center;
  justify-content: center;
  width: 180px;
  transition: all .5s;
  color: inherit;
  color: #2BA84A;
}

.pointer:hover {
  color: #42b983;
  border-color: #42b983;
}

.content {
  padding: 30px;  
  flex: 1;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  align-items: stretch;
  align-content: stretch;
  color: #42b983;
}

.input-window {
  width: 100%;
  display: flex;
  flex-direction: row;
  justify-content: space-around;
}

@media screen and (max-width: 600px) {
  .input-window  {
    flex-direction: column;
  }
}

.results {
  border: 1px solid;
  margin-top: 30px;
  padding: 5px;
  color: #2BA84A;
}

.output-window {
  line-height: 5px;
}

.inference-window {
  flex: 0;
  height: 100%;
  width: 100%;
  display: flex;
  flex-direction: column;
  justify-content: space-around;
}

.input-file {
  display: none !important;
}

.file-upload {
  max-width: 50%;
}

@media screen and (max-width: 600px) {
  .file-upload  {
    order: -1;
    max-width: 100%;
    margin-bottom: 50px;
  }
}

.preview {
  width: 224px;
  height: 224px;
  object-fit: cover;
}

.model-info {
  text-align: left;
  max-width: 50%; 
}

@media screen and (max-width: 600px) {
  .model-info  {
    max-width: 100%; 
  }
}


.model-info p {
  padding: 0;
  margin: 0;
}
</style>
