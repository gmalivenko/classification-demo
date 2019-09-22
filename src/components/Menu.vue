<template>
  <div>
    <div class="logo">
    <router-link to="/">
      
      <img class="svg-img" src="../assets/neural-network.svg" />
    </router-link>
    </div>
    <p> The models sandbox </p>
  </div>
</template>

<script>
export default {
  name: 'HelloWorld',
  props: {
    msg: String
  }
}
</script>

<!-- Add "scoped" attribute to limit CSS to this component only -->
<style scoped>
.logo {
  text-align: center;
  padding: 10px;
}

.svg-img {
  width: 100px;
}

p {
  padding: 0;
  margin: 0;
  text-align: center;
}
</style>
